// include/entity.h
#ifndef RL2_ENTITY_H_
#define RL2_ENTITY_H_
#include "game_math.h"
#include "constants.h"
#include <array>
#include <cstdint>

namespace rl2{

struct Size {
	uint32_t height;
	uint32_t width;
};


struct Stats{
	uint32_t health;
	float movement_speed;
	Size size;
};


class Entity{
public: 
	Vector2D position;
	Vector2D velocity;
	Stats stats;

};

class Player : public Entity{
};


struct Enemy {
	std::array<Vector2D, kNumEnemies> position;
	std::array<Vector2D, kNumEnemies> velocity;
	std::array<uint32_t, kNumEnemies> health;
	std::array<float, kNumEnemies> movement_speed;
	std::array<Size, kNumEnemies> size;
};

}// namespace rl2

#endif 
