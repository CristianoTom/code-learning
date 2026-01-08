import pygame 
from pygame.sprite import Group
from settings import Settings
from ship import Ship
from game_stats import Gamestats
import game_functions as gf

def run_game(): 
    # 初始化游戏并创建一个屏幕对象
    pygame.init()
    ai_settings = Settings()
    screen = pygame.display.set_mode((ai_settings.screen_width,
                                       ai_settings.screen_height)) 
    pygame.display.set_caption("Alien Invasion") 

    stats = Gamestats(ai_settings)        # 创建一个用于存储游戏统计信息的实例
    ship = Ship(ai_settings, screen)      # 创建一艘飞船实例
    bullets = Group()                     # 创建一个用于存储子弹的编组
    aliens = Group()                      # 创建一个用于存储外星人的编组

    gf.create_fleet(ai_settings, screen, ship, aliens)  # 创建外星人群

    # 开始游戏的主循环
    while True:                  
        gf.check_events(ai_settings, screen, ship, bullets)                    # 监视键盘和鼠标事件
        if stats.game_active:
            ship.update()                     # 更新飞船的位置
            gf.update_bullet(ai_settings, screen, ship, aliens, bullets)         # 更新子弹位置
            gf.update_aliens(ai_settings, stats, screen, ship, aliens, bullets)   
        gf.update_screen(ai_settings, screen, ship, aliens, bullets)         # 更新屏幕上的图像  
run_game()