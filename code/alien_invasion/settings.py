class Settings:
    '''存储《外星人入侵》的所有设置的类'''
    def __init__(self):
        '''初始化屏幕设置'''    # 初始化屏幕设置
        self.screen_width = 900
        self.screen_height = 600
        self.bg_color = (230, 230, 230)  
        self.ship_speed_factor = 0.5  # 飞船的设置

        # 子弹设置
        self.bullet_speed_factor = 2
        self.bullet_width = 300
        self.bullet_height = 15
        self.bullet_color = (60, 60, 60)
        self.bullets_allowed = 3

        # 外星人设置
        self.alien_speed_factor = 0.3
        self.fleet_drop_speed = 10
        self.fleet_direction = 1  # 1表示向右移，-1表示向左移


        