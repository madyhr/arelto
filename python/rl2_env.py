# python/rl2_env.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

import rl2_py

game = rl2_py.Game()

if game.initialize():
    print("Game initialized!")
    game.run()
    game.shutdown()
    print("Game was shut down.")
