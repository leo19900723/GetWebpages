import os
from selenium.common.exceptions import NoSuchElementException
from Libs.CommonFunctions import webdriver_initiate
from Libs.CommonFunctions import pass_ptt_age_block


def push_and_dislike_popularity_translator(target_str):
    if target_str == "":
        return 0
    elif target_str == "爆":
        return 100
    elif target_str == "XX":
        return -100
    elif target_str[0] == "X":
        return int(target_str[1]) * -10
    else:
        return int(target_str)


# Time complexity: Omega(m*n)
def get_ptt_target_titles(driver, board_list, articles_num, popularity_filter):
    output = []

    for boards_name in board_list:
        driver.get(str("https://www.ptt.cc/bbs/" + boards_name + "/index" + ".html"))

        temp_output = []
        temp_output.append(boards_name)
        # Initiate counter
        counter = articles_num

        while counter > 0:
            for element in (x for x in pass_ptt_age_block(driver).find_elements_by_xpath("//div[@class='r-ent']") if counter > 0):
                if push_and_dislike_popularity_translator(element.find_element_by_class_name("nrec").text) >= popularity_filter[0] and push_and_dislike_popularity_translator(element.find_element_by_class_name("nrec").text) <= popularity_filter[1]:
                    try:
                        temp_output.append(element.find_element_by_tag_name("a").get_attribute("href"))
                    except NoSuchElementException:
                        continue
                    else:
                        counter -= 1
            driver.find_elements_by_xpath("//a[@class='btn wide']")[1].click()

        output.append(temp_output)

    return output


def unittest_main():
    test_driver = webdriver_initiate()
    test_popularity_filter = [-100, -20]  # The first digit indicate the lower bound of push of dislike value. The second digit indicate the higher bound. The result will include the setting boundary.
    test_articles_num = 10  # This variable will control that how many pages will be parsed before the latest page.
    test_board_list = ["Gossiping", "sex", "HatePolitics"]  # A list that indicates which PTT-Board will be parsed

    screen_or_doc = "doc"

    # Create a directory to restore title lists.
    if not os.path.isdir("..\\Results"):
        os.mkdir("..\\Results\\")

    for i in get_ptt_target_titles(test_driver, test_board_list, test_articles_num, test_popularity_filter):
        if screen_or_doc == "screen":
            print("\n".join(i[1:]))
        else:
            with open(str("..\\Results\\" + i[0] + ".csv"), "w") as output_title:
                output_title.write("\n".join(i[1:]))

    test_driver.close()


if __name__ == "__main__":
    unittest_main()
