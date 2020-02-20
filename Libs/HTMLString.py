
class HTMLString:

    def __init__(self, str_list):
        self.__sourcecode_list = str_list

    @classmethod
    def from_file(cls, html_file_name, assigned_encoding):
        source_file = open(html_file_name, "r", encoding=assigned_encoding)
        output_list = []
        for element in source_file:
            if not element.isspace():
                output_list.append(element.strip())
        return cls(output_list)

    def get_source(self):
        return self.__sourcecode_list

    def get_text(self, index=0):
        output_str = ""
        allow_to_print_tag = True
        for element in self.__sourcecode_list[index]:
            if element == "<":
                allow_to_print_tag = False
            elif element == ">":
                allow_to_print_tag = True
            elif allow_to_print_tag:
                output_str += element
        return output_str.strip()

