class Exception(BaseException):
    def __init__(self,mes):
        self.mes = mes
    def __str__(self):
        return self.mes

