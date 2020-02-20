import time
import unicodedata
from Libs.CommonFunctions import *


def push_and_dislike_time_translator(time_string):
    if len(time_string) <= 11:
        return (int(time_string.split(" ")[0].split("/")[0]) * 30 + int(time_string.split(" ")[0].split("/")[1])) * 1440 + int(time_string.split(" ")[1].split(":")[0]) * 60 + int(time_string.split(" ")[1].split(":")[1])
    else:
        return (int(time_string.split(" ")[1].split("/")[0]) * 30 + int(time_string.split(" ")[1].split("/")[1])) * 1440 + int(time_string.split(" ")[2].split(":")[0]) * 60 + int(time_string.split(" ")[2].split(":")[1])


def push_and_dislike_max_words_indicator(string, iptime_string_len):
    length = 0
    for char in string:
        length += 2 if unicodedata.east_asian_width(char) != "Na" else 1
    return True if length >= (76 if iptime_string_len == 11 else 75) else False


def punctuation_detector(argument):
    return {
        ",": True,
        "，": True,
        ".": True,
        "。": True,
        "!": True,
        "！": True,
        "?": True,
        "？": True,
        "~": True,
        "～": True,
        "'": True,
        "、": True,
        "(": True,
        "（": True,
        ")": True,
        "）": True,
        ";": True,
        "；": True
    }.get(argument, None)


# Time complexity: O(2m*log n)
def get_ptt_target_articles(driver, prefix, time_slot_for_merging):
    output_result = []

    # Get the main article. Split each line by token: <br> then store the whole article into the first entry of output list.
    main_content = pass_ptt_age_block(driver).find_element_by_xpath("//div[@id='main-content']")
    temp_result = [prefix + "_" + f'{0:05}', str(main_content.find_element_by_class_name("article-meta-value").text).split(" ")[0], "", ""]
    for element in str(main_content.text).splitlines(False)[4:]:
        temp_result[2] += str(element + "<br>")
        if element == "--":  # "--" is the EOF notation in ptt which indicates the end of the main article.
            output_result.append(temp_result)
            break

    print("Get push and dislike comments.")
    measure_start = time.time()
    # Get every push of dislike response related to the article.
    comments_content = driver.find_elements_by_xpath("//div[@class='push']")
    for element in comments_content:
        temp_result = ["", "", "", "", 0]
        temp_result[parsedOutPutHeadersDict["User_ID"]] = element.find_element_by_class_name("push-userid").text  # Store the ID
        temp_result[parsedOutPutHeadersDict["Main_Content"]] = str(element.find_element_by_class_name("push-content").text).replace(":", "").strip()  # Store the comment
        temp_result[parsedOutPutHeadersDict["Time"]] = str(element.find_element_by_class_name("push-ipdatetime").text).strip()  # Store the date/ time that the comment response (and IP address if any)
        temp_result[parsedOutPutHeadersDict["Max_Word_Indicator"]] = push_and_dislike_max_words_indicator(str(element.text).replace("\n", "").strip(), len(temp_result[2]))  # Detect the number of words of the comment has reached the maximum limit or not. "True" means: it did reach the maximum limit.
        output_result.append(temp_result)
    print("\tThe time cost of this loop:", time.time() - measure_start)

    # Merge the comment by time slot and ID
    index = 1
    while index <= len(output_result[1:]):
        output_result[index][parsedOutPutHeadersDict["Serial_Number"]] = prefix + "_" + f'{index:05}'
        sub_index = index + 1
        max_words_indicator = output_result[index][parsedOutPutHeadersDict["Max_Word_Indicator"]]
        while sub_index <= len(output_result[1:]) and push_and_dislike_time_translator(output_result[sub_index][parsedOutPutHeadersDict["Time"]]) - push_and_dislike_time_translator(output_result[index][parsedOutPutHeadersDict["Time"]]) <= time_slot_for_merging:
            if output_result[sub_index][parsedOutPutHeadersDict["User_ID"]] == output_result[index][parsedOutPutHeadersDict["User_ID"]]:
                output_result[index][parsedOutPutHeadersDict["Main_Content"]] += str(("，" if not (punctuation_detector(output_result[index][parsedOutPutHeadersDict["Main_Content"]][len(output_result[index][parsedOutPutHeadersDict["Main_Content"]]) - 1]) or max_words_indicator) else "") + output_result[sub_index][parsedOutPutHeadersDict["Main_Content"]])
                max_words_indicator = output_result[sub_index][parsedOutPutHeadersDict["Max_Word_Indicator"]]
                del output_result[sub_index]
            else:
                sub_index += 1
        index += 1

    return output_result


def __unittest_main():
    test_url = "https://www.ptt.cc/bbs/Gossiping/M.1556536524.A.563.html"
    test_file_name = test_url.replace("https://www.ptt.cc/bbs/", "").replace(".html", "").replace("/", "_")
    test_time_slot_for_merging = 1

    print("Start the webdriver module")
    measure_start = time.time()
    test_driver = webdriver_initiate(os.path.abspath(""))
    test_driver.get(test_url)
    print("Finish the initiation of webdriver module, time cost:", time.time() - measure_start)

    screen_or_doc = "screen"

    with open(str(test_file_name + ".csv"), "w", encoding="utf-8") as parsed_article_file:
        for index, element in enumerate(get_ptt_target_articles(test_driver, test_file_name, test_time_slot_for_merging)):
            if screen_or_doc == "screen":
                print(index, element)
            else:
                parsed_article_file.write(str("\"" + element[parsedOutPutHeadersDict["Serial_Number"]] + "\", \"" + element[parsedOutPutHeadersDict["User_ID"]] + "\", \"" + element[parsedOutPutHeadersDict["Main_Content"]] + "\"\n"))

    test_driver.close()


if __name__ == "__main__":
    __unittest_main()
