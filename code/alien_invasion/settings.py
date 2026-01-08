class Settings:
    '''存储《外星人入侵》的所有设置的类'''
    def __init__(self):
        '''初始化屏幕设置'''    # 初始化屏幕设置
        self.screen_width = 900
        self.screen_height = 600
        self.bg_color = (230, 230, 230)  
        self.ship_speed_factor = 1.5  # 飞船的设置
        