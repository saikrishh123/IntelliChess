import numpy as np
import cv2 as cv
import chess
from stockfish import Stockfish
import win32api, win32con
from time import sleep
from chess import Move
import pyautogui
import utils
CLICK_GAP=0.1

class game():
    def __init__(self, gcords,stock_fish_path,depth=15):
        self.gcords=gcords
        self.board=chess.Board()
        self.board.clear_board()
        self.engine=Stockfish(stock_fish_path,depth=depth)
        self.map_dic=self.create_map_dic()
        self.gx=gcords[0]
        self.gy=gcords[1]
        self.gm=gcords[2]
        

    
    def create_map_dic(self):
        d=dict()
        cnt=0;
        for i in range(8):
            for j in range(8):
                d[cnt]=(j,i)
                cnt+=1
        return d
    
    def click(self,x,y):
        win32api.SetCursorPos((x,y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    

    def compute_centre(self,square):
        i,j=self.map_dic[square]
        xn=int(self.gx+i*self.gm+self.gm/2)
        yn=int(self.gy+(7-j)*self.gm+self.gm/2)
        return xn,yn
                
    def move(self,uci):
        sleep(1)
        mv=Move.from_uci(uci)
        fs=mv.from_square
        ts=mv.to_square
        x,y=self.compute_centre(fs)
        self.click(x,y)
        sleep(CLICK_GAP)
        x,y=self.compute_centre(ts)
        self.click(x,y)

    
    def update_board(self):
        sleep(1)
        imgp=np.array(pyautogui.screenshot())
        imgp = cv.cvtColor(np.array(imgp), cv.COLOR_RGB2BGR)
        self.img=imgp
        self.board=utils.create_board(imgp,self.gx,self.gy,self.gm)
    
    
    def autoPlay(self):
        self.board=self.update_board()
        self.engine.set_fen_position(self.board.fen())
        mv=self.engine.get_best_move()
        self.move(self,mv)
        self.board=self.update_board()
