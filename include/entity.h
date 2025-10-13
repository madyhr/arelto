// include/entity.h
#ifndef RL2_ENTITY_H_
#define RL2_ENTITY_H_
#include "game_math.h"
#include <cstdint>

namespace rl2{

struct Size {
	uint32_t height;
	uint32_t width;
};


struct Stats{
	uint32_t health;
	uint32_t movement_speed;
	Size size;
};


struct Entity {
	Vector2D position;
	Vector2D velocity;
	Stats stats;
};

} // namespace rl2

#endif 
