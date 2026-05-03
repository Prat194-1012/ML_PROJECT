
import sys 

def error_message_details(error, error_details):
    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message: str='error occured in python script name [{0}] line number [{1}] error message [{2}]'.format(file_name, line_number,str(error))
    return error_message

class custom_exception(Exception):
    def __init__(self, error, error_details):
        super().__init__(error_message_details(error, error_details))
        self.error_message_details = error_message_details(error, error_details=error_details)

    def __str__(self):
        return self.error_message

