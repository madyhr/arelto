import torch
import torch.nn as nn


class RayEncoder(nn.Module):
    def __init__(
        self,
        num_rays: int,
        num_ray_types: int,
        output_dim: int,
        history_length: int = 1,
        embedding_dim: int = 8,
        out_channels_list: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        strides: list[int] | None = None,
        activation_func_class: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if out_channels_list is None:
            out_channels_list = [16, 32]
        if kernel_sizes is None:
            kernel_sizes = [3, 3]
        if strides is None:
            strides = [1, 1]

        assert len(out_channels_list) == len(kernel_sizes) == len(strides)

        self.num_rays = num_rays
        self.num_ray_types = num_ray_types
        self.output_dim = output_dim
        self.history_length = history_length

        self.type_embed = nn.Embedding(num_ray_types, embedding_dim)

        # 1 for distance and the rest depends on embedding dim
        # We multiply by history_length because we stack frames channel-wise
        self.num_ray_channels = 1 + embedding_dim
        in_channels = history_length * self.num_ray_channels
        self.total_rays = num_rays * history_length

        layers = []
        current_channels = in_channels

        for out_channels, kernel_size, stride in zip(
            out_channels_list, kernel_sizes, strides
        ):
            layers.append(
                nn.Conv1d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    padding_mode="circular",
                )
            )
            layers.append(activation_func_class())
            current_channels = out_channels

        self.conv_net = nn.Sequential(*layers)

        # Calculate flattened dimension to define the final linear layer
        # require a dummy pass to determine size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, num_rays)
            dummy_output = self.conv_net(dummy_input)
            self.flatten_dim = dummy_output.view(1, -1).shape[1]

        self.fc = nn.Linear(self.flatten_dim, output_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # B = Number of enemies (batch size/num envs)
        # T = history length
        # N = number of rays
        # E = embedding dim

        batch_size = obs.shape[0]

        # (B, T, N)
        ray_distances = obs[:, : self.total_rays].view(
            batch_size, self.history_length, self.num_rays
        )
        ray_types = (
            obs[:, self.total_rays :]
            .view(batch_size, self.history_length, self.num_rays)
            .long()
        )

        # (B, T, N) -> (B, T, 1, N)
        ray_distances = ray_distances.unsqueeze(2)

        # (B, T, N) -> (B, T, N, E) -> (B, T, E, N)
        embedded_types = self.type_embed(ray_types).permute(0, 1, 3, 2)

        # (B, T, 1+E, N)
        combined = torch.cat([ray_distances, embedded_types], dim=2)

        # (B, T * (1+E), N)
        combined = combined.flatten(1, 2)

        conv_out = self.conv_net(combined)
        flat_out = conv_out.view(batch_size, -1)

        return self.fc(flat_out)
