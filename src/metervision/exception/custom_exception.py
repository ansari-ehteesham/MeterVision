import sys

from metervision.logger.logs import logging


def cutome_error_message(error: str, error_details: sys):
    _, _, exc_tb = error_details.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        "Error Occured in Python Script Name [{0}] "
        "line number [{1}] error message [{2}]".format(
            file_name, line_number, str(error)
        )
    )

    logging.error(error_message)
    return error_message


class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = cutome_error_message(error_message, error_detail)

    def __str__(self):
        return self.error_message
