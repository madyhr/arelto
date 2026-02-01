// include/constants/map.h
#ifndef RL2_CONSTANTS_MAP_H_
#define RL2_CONSTANTS_MAP_H_
#include <cmath>

namespace arelto {
// Map constants
constexpr int kMapWidth = 10000;
constexpr int kMapHeight = 10000;
// the inverse map max distance is used for scaling distances.
const float kInvMapMaxDistance =
    1.0f / std::sqrt(static_cast<float>(kMapHeight) * kMapHeight +
                     static_cast<float>(kMapWidth) * kMapWidth);
constexpr int kOccupancyMapResolution = 25;
constexpr int kOccupancyMapWidth =
    (kMapWidth + kOccupancyMapResolution - 1) / kOccupancyMapResolution;
constexpr int kOccupancyMapHeight =
    (kMapHeight + kOccupancyMapResolution - 1) / kOccupancyMapResolution;
// number of physics steps per occupancy map update
constexpr int kOccupancyMapTimeDecimation = 100;
// Tiles are used for rendering
constexpr int kTileWidth = 40;
constexpr int kTileHeight = 119;
constexpr int kNumTileTypes = 44;
constexpr int kNumTilesX = ((kMapWidth + kTileWidth - 1) / kTileWidth);
constexpr int kNumTilesY = ((kMapHeight + kTileHeight - 1) / kTileHeight);
}  // namespace arelto
#endif
