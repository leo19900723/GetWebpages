from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from Libs.CommonFunctions import *

__general_timeout = 5
__quick_timeout = 3


# Time complexity: O(m*n)
def get_ptt_target_titles(driver, username, password, board_list, articles_num, popularity_filter, saved_url_list=None):
    output_result = []
    saved_url_dict = {}
    try:
        saved_url_dict = dict(zip(saved_url_list, range(len(saved_url_list))))
    except TypeError:
        pass

    # Log into PTT
    WebDriverWait(driver, __general_timeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='20']"), "請輸入代號，或以 guest 參觀，或以 new 註冊:                                     "))
    driver.find_element_by_id("t").send_keys(str(username + "\r\n" + password + "\r\n"))

    # Multiple log in status handling.
    try:
        WebDriverWait(driver, __general_timeout).until(EC.presence_of_element_located((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='21']/span/span[@class='q15 b0']")))
        if driver.find_element_by_xpath("//span[@class='' and @data-type='bbsline' and @data-row='21']/span/span[@class='q15 b0']").text == "注意: 您有其它連線已登入此帳號。":
            driver.find_element_by_id("t").send_keys(str("n\r\n"))
    except TimeoutException:
        print("No doubling login needed to be handled.")

    # Welcome page when log in PTT handling.
    try:
        WebDriverWait(driver, __quick_timeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='23']"), " ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ 請按任意鍵繼續 ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄  "))
        driver.find_element_by_id("t").send_keys(str("\r\n"))
    except TimeoutException:
        print("No welcome AD needed to be handled.")

    for target_boards in board_list:
        temp_result = [target_boards]

        # Search for target boards
        WebDriverWait(driver, __general_timeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='12']"), "───────── 上方為使用者心情點播留言區，不代表本站立場 ────────  "))
        driver.find_element_by_id("t").send_keys(str("s" + target_boards + "\r\n"))

        # Welcome page when entering target boards handling.
        try:
            WebDriverWait(driver, __quick_timeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='23']"), " ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ 請按任意鍵繼續 ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄  "))
            driver.find_element_by_id("t").send_keys(str("\r\n"))
        except TimeoutException:
            print("No Board's welcome AD needed to be handled.")

        # Filtering specific popularity
        driver.find_element_by_id("t").send_keys(str("Z" + popularity_filter + "\r\n"))
        try:
            WebDriverWait(driver, __general_timeout).until(EC.presence_of_element_located((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='0']/span/span[@class='q10 b4']")))

            # Locate starting row according to the filtering result
            row_counter = len(str(driver.find_element_by_id("mainContainer").text).splitlines(True)) - 2
            # If the number of the filtering result is smaller than the assigned article number, every filtering result will be printed and quite the loop.
            recorded_articles_counter = 0 if int(str(driver.find_element_by_xpath("//span[@class='' and @data-type='bbsline' and @data-row='" + str(row_counter) + "']/span/span[@class='q7 b0']").text).strip("●").strip().split(" ")[0]) >= articles_num else articles_num - int(str(driver.find_element_by_xpath("//span[@class='' and @data-type='bbsline' and @data-row='" + str(row_counter) + "']/span/span[@class='q7 b0']").text).strip("●").strip().split(" ")[0])
            # Get articles list.
            while recorded_articles_counter < articles_num:
                driver.find_element_by_id("t").send_keys(str("Q"))
                try:
                    WebDriverWait(driver, __quick_timeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='" + str((row_counter - 3) if row_counter >= 15 else (row_counter + 3)) + "']/span/span[@class='q7 b0']"), "│ 文章網址: "))
                    temp_url = str(driver.find_element_by_xpath(str("//span[@class='' and @data-type='bbsline' and @data-row='" + str((row_counter - 3) if row_counter >= 15 else (row_counter + 3)) + "']/a")).get_attribute("href"))
                    try:
                        saved_url_dict[temp_url]
                    except KeyError:
                        temp_result.append(temp_url)
                    recorded_articles_counter += 1
                except TimeoutException:
                    print("The article has been deleted.")

                driver.find_element_by_id("t").send_keys("\r\n" + Keys.ARROW_UP)
                if row_counter > 3:
                    row_counter -= 1
                else:
                    driver.find_element_by_id("t").send_keys(Keys.ARROW_DOWN)
                    row_counter = 22
            output_result.append(temp_result)
            driver.find_element_by_id("t").send_keys(Keys.ARROW_LEFT)
        except TimeoutException:
            print("According to your popularity filtering setting, no related result can be located.")

        WebDriverWait(driver, __general_timeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='23']/span/span[@class='q4 b6']"), " 文章選讀 "))
        driver.find_element_by_id("t").send_keys(Keys.ARROW_LEFT)

    return output_result


def __unittest_main():
    test_ptt_user_name = input("Please enter your PTT account: ")
    test_ptt_user_password = input("Please enter your passwords: ")

    test_driver = webdriver_initiate(os.path.abspath(""))
    test_driver.get("https://term.ptt.cc/")

    test_popularity_filter = "-100"
    test_board_list = ["Gossiping", "sex", "HatePolitics"]
    test_articles_num = 5

    screen_or_doc = "doc"
    
    # Create a directory to restore title lists.
    if not os.path.isdir(".\\Results"):
        os.mkdir(".\\Results\\")

    with open(str(".\\Results\\" + "Articles_list.csv"), "w") as output_title:
        output_title.write(str("\"" + "\", \"".join(articleListHeadersList[:2]) + "\"\n"))
        for board in get_ptt_target_titles(test_driver, test_ptt_user_name, test_ptt_user_password, test_board_list, test_articles_num, test_popularity_filter):
            for url in board[1:]:
                print_line = "\"" + board[0] + "\", \"" + url + "\"\n"
                if screen_or_doc == "screen":
                    print(print_line)
                else:
                    output_title.write(print_line)

    test_driver.close()


if __name__ == "__main__":
    __unittest_main()
