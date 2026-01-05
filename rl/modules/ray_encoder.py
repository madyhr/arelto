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
        in_channels = history_length * (1 + embedding_dim)

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
        # Obs layout: [All Distances (T*N), All Types (T*N)]
        # T = history_length
        # N = num_rays

        total_rays = self.num_rays * self.history_length
        ray_distances_flat = obs[:, :total_rays]
        ray_types_flat = obs[:, total_rays:].long()

        # Reshape to (B, T, N)
        # We assume the order in obs is [Frame0_Rays, Frame1_Rays, ...]
        batch_size = obs.shape[0]
        ray_distances = ray_distances_flat.view(
            batch_size, self.history_length, self.num_rays
        )  # (B, T, N)
        ray_types = ray_types_flat.view(
            batch_size, self.history_length, self.num_rays
        )  # (B, T, N)

        embedded_types = self.type_embed(ray_types)  # (B, T, N, E)
        ray_distances = ray_distances.unsqueeze(-1)  # (B, T, N, 1)

        # Concatenate: (B, T, N, 1 + E)
        combined = torch.cat([ray_distances, embedded_types], dim=-1)

        # Reshape to stack channels: (B, N, T * (1 + E))
        # First permute to (B, N, T, 1+E)
        combined = combined.permute(0, 2, 1, 3)
        # Then flatten the last two dims
        combined = combined.reshape(batch_size, self.num_rays, -1)

        # Permute for Conv1d: (B, Channels, N)
        combined = combined.permute(0, 2, 1).contiguous()

        conv_out = self.conv_net(combined)
        flat_out = conv_out.view(conv_out.size(0), -1)

        return self.fc(flat_out)
