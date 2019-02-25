# -*- coding: utf-8 -*-


"""
Created on Fri Feb  8 15:17:36 2019

@author: Sai Krishna
"""


# Python 2/3 compatibility
import play

    
gx=310
gy=166
gm=91
stock_fish_path='C:\CodeFiles\stockfish-10-win\stockfish-10-win\Windows\stockfish_10_x64'
g=play.game((gx,gy,gm),stock_fish_path)

n_moves=30 
for i in range(n_moves):
    g.update_board()   
    g.engine.set_fen_position(g.board.fen())
    mv=g.engine.get_best_move()
    g.move(mv)
        
        
        

        
    













