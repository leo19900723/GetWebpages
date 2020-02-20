import os
import time
import unicodedata
import threading
from selenium import webdriver
from selenium.common.exceptions import SessionNotCreatedException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys


class PttDataQuery(object):

    def __init__(self, userName, userPassword, boardList, articleNum, popularityFilter):
        self.__pttConsoleDriver = self.__webdriverInitiate(os.path.abspath(""))
        self.__pttWebDriver = self.__webdriverInitiate(os.path.abspath(""))
        self.__userName = userName
        self.__userPassword = userPassword
        self.__filePath = ".\\Results\\"
        self.__articleNum = articleNum
        self.__popularityFilter = popularityFilter
        self.__generalTimeout = 5
        self.__quickTimeout = 3
        self.__timeSlotForMerging = 1

        self.__boardList = boardList
        self.__boardListDict = dict((entry, "") for entry in self.__boardList)
        self.__titleUrlList = []
        self.__titleUrlListDict = {}
        self.__parsedArticleList = []

        self.__titleUrlHeadersList = ["Board_Name", "Article_URL"]
        self.__parsedArticleHeadersList = ["Serial_Number", "User_ID", "Main_Content", "Time", "Max_Word_Indicator"]
        self.__titleUrlHeadersDict = dict(zip(self.__titleUrlHeadersList, range(len(self.__titleUrlHeadersList))))
        self.__parsedArticleHeadersDict = dict(zip(self.__parsedArticleHeadersList, range(len(self.__parsedArticleHeadersList))))

        self.__punctuationDict = dict((value, True) for value in [",", "，", ".", "。", "!", "！", "?", "？", "~", "～", "'", "、", "(", "（", ")", "）", ";", "；"])

        self.__pttConsoleDriver.get("https://term.ptt.cc/")
        self.__passPttLoginSessions()

    def __webdriverInitiate(self, hooker_directory_path):
        # Firefox x86 handling
        try:
            return webdriver.Firefox(executable_path=str(hooker_directory_path + "\\WebDriver\\geckodriver_x86.exe"))

        # Firefox x64 handling
        except SessionNotCreatedException:
            return webdriver.Firefox(executable_path=str(hooker_directory_path + "\\WebDriver\\geckodriver_x64.exe"))

        # Google Chrome Handling
        except:
            return webdriver.Chrome(executable_path=str(hooker_directory_path + "\\WebDriver\\chromedriver.exe"))

    def __passPttAgeBlock(self):
        while True:
            try:  # Handling "age-limited" warning page. Use HTML's attributions to locate a specific block (in this case: submitting-button), then proceed relative actions: clicking "YES".
                warning_submitting_button = self.__pttWebDriver.find_elements_by_xpath("//button[@type='submit' and @name='yes']")[0]
                warning_submitting_button.click()
                return self.__pttWebDriver
            except (AttributeError, IndexError):  # Get target-page's source code
                return self.__pttWebDriver

    def __passPttLoginSessions(self):
        # Log into PTT
        WebDriverWait(self.__pttConsoleDriver, self.__generalTimeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='20']"), "請輸入代號，或以 guest 參觀，或以 new 註冊:                                     "))
        self.__pttConsoleDriver.find_element_by_id("t").send_keys(str(self.__userName + "\r\n" + self.__userPassword + "\r\n"))

        # Multiple log in status handling.
        try:
            WebDriverWait(self.__pttConsoleDriver, self.__generalTimeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='21']/span/span[@class='q15 b0']"), "注意: 您有其它連線已登入此帳號。"))
            self.__pttConsoleDriver.find_element_by_id("t").send_keys(str("n\r\n"))
        except TimeoutException:
            print("No doubling login needed to be handled.")

        # Welcome page when log in PTT handling.
        try:
            WebDriverWait(self.__pttConsoleDriver, self.__quickTimeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='23']"), " ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ 請按任意鍵繼續 ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄  "))
            self.__pttConsoleDriver.find_element_by_id("t").send_keys(str("\r\n"))
        except TimeoutException:
            print("No welcome AD needed to be handled.")


    def __pushDislikeTimeTranslator(self, timeString):
        try:
            if len(timeString) <= 11:
                return (int(timeString.split(" ")[0].split("/")[0]) * 30 + int(timeString.split(" ")[0].split("/")[1])) * 1440 + int(timeString.split(" ")[1].split(":")[0]) * 60 + int(timeString.split(" ")[1].split(":")[1])
            else:
                return (int(timeString.split(" ")[1].split("/")[0]) * 30 + int(timeString.split(" ")[1].split("/")[1])) * 1440 + int(timeString.split(" ")[2].split(":")[0]) * 60 + int(timeString.split(" ")[2].split(":")[1])
        except:
            return 0

    def __pushDislikeMaxWordsIndicator(self, string, iptimeStringLen):
        length = 0
        for char in string:
            length += 2 if unicodedata.east_asian_width(char) != "Na" else 1
        return "1" if length >= (76 if iptimeStringLen == 11 else 75) else "0"

    def setUserName(self, newUserName):
        self.__userName = newUserName

    def setUserPassword(self, newUserPassword):
        self.__userPassword = newUserPassword

    def setBoardList(self, newBoardList):
        del self.__boardList[:]
        self.__boardList = newBoardList

    def setArticleNum(self, newArticleNum):
        self.__articleNum = newArticleNum

    def setPopularityFilter(self, newPopularityFilter):
        self.__popularityFilter = newPopularityFilter

    def getPttTitleUrl(self, targetBoard=None):
        if targetBoard is not None:
            if self.__boardListDict[targetBoard] == "":
                self.__titleUrlList += self.__getPttTitleUrlCore(targetBoard)
            filteredList = []
            for index in self.__boardListDict[targetBoard].split(",")[1:]:
                filteredList.append(self.__titleUrlList[int(index)])
            return filteredList
        else:
            for boards in self.__boardList:
                if self.__boardListDict[boards] == "":
                    self.__titleUrlList += self.__getPttTitleUrlCore(boards)
            return self.__titleUrlList

    def __getPttTitleUrlCore(self, targetBoard):
        outputUrlList = []

        # Search for target boards
        WebDriverWait(self.__pttConsoleDriver, self.__generalTimeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='12']"), "───────── 上方為使用者心情點播留言區，不代表本站立場 ────────  "))
        self.__pttConsoleDriver.find_element_by_id("t").send_keys(str("s" + targetBoard + "\r\n"))

        # Welcome page when entering target boards handling.
        try:
            WebDriverWait(self.__pttConsoleDriver, self.__quickTimeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='23']"), " ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ 請按任意鍵繼續 ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄  "))
            self.__pttConsoleDriver.find_element_by_id("t").send_keys(str("\r\n"))
        except TimeoutException:
            print("No Board: " + targetBoard + "'s welcome AD needed to be handled.")

        # Filtering specific popularity
        self.__pttConsoleDriver.find_element_by_id("t").send_keys(str("Z" + self.__popularityFilter + "\r\n"))
        self.__pttConsoleDriver.find_element_by_id("t").send_keys(Keys.END)
        try:
            WebDriverWait(self.__pttConsoleDriver, self.__generalTimeout).until(EC.presence_of_element_located((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='0']/span/span[@class='q10 b4']")))

            # Locate starting row according to the filtering result
            rowCounter = len(str(self.__pttConsoleDriver.find_element_by_id("mainContainer").text).splitlines(True)) - 2
            # If the number of the filtering result is smaller than the assigned article number, every filtering result will be printed and quite the loop.
            recordedArticlesCounter = 0 if int(str(self.__pttConsoleDriver.find_element_by_xpath("//span[@class='' and @data-type='bbsline' and @data-row='" + str(rowCounter) + "']/span/span[@class='q7 b0']").text).strip("●").strip().split(" ")[0]) >= self.__articleNum else self.__articleNum - int(str(self.__pttConsoleDriver.find_element_by_xpath("//span[@class='' and @data-type='bbsline' and @data-row='" + str(rowCounter) + "']/span/span[@class='q7 b0']").text).strip("●").strip().split(" ")[0])
            # Get articles list.
            while recordedArticlesCounter < self.__articleNum:
                self.__pttConsoleDriver.find_element_by_id("t").send_keys(str("Q"))
                try:
                    WebDriverWait(self.__pttConsoleDriver, self.__quickTimeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='" + str((rowCounter - 3) if rowCounter >= 15 else (rowCounter + 3)) + "']/span/span[@class='q7 b0']"), "│ 文章網址: "))
                    tempUrl = str(self.__pttConsoleDriver.find_element_by_xpath(str("//span[@class='' and @data-type='bbsline' and @data-row='" + str((rowCounter - 3) if rowCounter >= 15 else (rowCounter + 3)) + "']/a")).get_attribute("href"))
                    try:
                        self.__titleUrlListDict[tempUrl]
                    except KeyError:
                        tempUrlList = [None] * len(self.__titleUrlHeadersList)
                        tempUrlList[self.__titleUrlHeadersDict["Board_Name"]] = tempUrl.split("/")[4]
                        tempUrlList[self.__titleUrlHeadersDict["Article_URL"]] = tempUrl
                        outputUrlList.append(tempUrlList)
                        self.__titleUrlListDict[tempUrl] = len(self.__titleUrlListDict)
                        self.__boardListDict[targetBoard] += "," + str(self.__titleUrlListDict[tempUrl])
                        recordedArticlesCounter += 1
                except TimeoutException:
                    print("The article has been deleted.")

                self.__pttConsoleDriver.find_element_by_id("t").send_keys("\r\n" + Keys.ARROW_UP)
                if rowCounter > 3:
                    rowCounter -= 1
                else:
                    self.__pttConsoleDriver.find_element_by_id("t").send_keys(Keys.ARROW_DOWN)
                    rowCounter = 22
            self.__pttConsoleDriver.find_element_by_id("t").send_keys(Keys.ARROW_LEFT)
        except TimeoutException:
            print("According to your popularity filtering setting, no related result can be located.")

        WebDriverWait(self.__pttConsoleDriver, self.__generalTimeout).until(EC.text_to_be_present_in_element((By.XPATH, "//span[@class='' and @data-type='bbsline' and @data-row='23']/span/span[@class='q4 b6']"), " 文章選讀 "))
        self.__pttConsoleDriver.find_element_by_id("t").send_keys(Keys.ARROW_LEFT)
        return outputUrlList

    def getPttParsedArticle(self, requeryFlag=True):
        boardFunc = lambda urlString: urlString.split("/")[4]
        articleUniqueIDFunc = lambda urlString: urlString.split("/")[5].replace(".html", "")

        if requeryFlag:
            self.__parsedArticleList = []
            for index, element in enumerate(self.__titleUrlList):
                print("Get #", index)
                try:
                    self.__pttWebDriver.get(element[self.__titleUrlHeadersDict["Article_URL"]])
                    threading.Thread(target=self.__parsedArticleList.append(self.__getPttParsedArticleCore(boardFunc(element[self.__titleUrlHeadersDict["Article_URL"]]) + "#" + articleUniqueIDFunc(element[self.__titleUrlHeadersDict["Article_URL"]]))), daemon=True)
                except:
                    continue
            return self.__parsedArticleList
        else:
            return self.__parsedArticleList

    def __getPttParsedArticleCore(self, prefix):
        outputList = []

        # Get the main article. Split each line by token: <br> then store the whole article into the first entry of output list.
        main_content = self.__passPttAgeBlock().find_element_by_xpath("//div[@id='main-content']")
        temp_result = [prefix + "#" + f'{0:05}', str(main_content.find_element_by_class_name("article-meta-value").text).split(" ")[0], "", "", ""]
        for element in str(main_content.text).splitlines(False)[4:]:
            temp_result[2] += str(element + "<br>")
            if element == "--":  # "--" is the EOF notation in ptt which indicates the end of the main article.
                outputList.append(temp_result)
                break

        print("Get push and dislike comments.")
        measure_start = time.time()
        # Get every push of dislike response related to the article.
        comments_content = self.__pttWebDriver.find_elements_by_xpath("//div[@class='push']")
        for element in comments_content:
            temp_result = [""]

            userID = element.find_element_by_class_name("push-userid").text
            mainContent = str(element.find_element_by_class_name("push-content").text).replace(":", "").strip()
            timeAndDate = str(element.find_element_by_class_name("push-ipdatetime").text).strip()

            if userID and mainContent and timeAndDate:
                getUserID = lambda x: x.insert(self.__parsedArticleHeadersDict["User_ID"], userID)
                getMainContent = lambda x: x.insert(self.__parsedArticleHeadersDict["Main_Content"], mainContent)
                getTime = lambda x: x.insert(self.__parsedArticleHeadersDict["Time"], timeAndDate)
                getMaxWordIndicator = lambda x: x.insert(self.__parsedArticleHeadersDict["Max_Word_Indicator"], self.__pushDislikeMaxWordsIndicator(str(element.text).replace("\n", "").strip(), len(x[self.__parsedArticleHeadersDict["Time"]])))

                threading.Thread(target=getUserID(temp_result), daemon=True).run()
                threading.Thread(target=getMainContent(temp_result), daemon=True).run()
                threading.Thread(target=getTime(temp_result), daemon=True).run()
                threading.Thread(target=getMaxWordIndicator(temp_result), daemon=True).run()

                outputList.append(temp_result)
        print("\tThe time cost of this loop:", time.time() - measure_start)

        # Merge the comment by time slot and ID
        index = 1
        while index <= len(outputList[1:]):
            if outputList[index][self.__parsedArticleHeadersDict["Main_Content"]] != "":
                outputList[index][self.__parsedArticleHeadersDict["Serial_Number"]] = prefix + "#" + f'{index:05}'
                sub_index = index + 1
                maxWordsIndicator = int(outputList[index][self.__parsedArticleHeadersDict["Max_Word_Indicator"]])
                while sub_index <= len(outputList[1:]) and self.__pushDislikeTimeTranslator(outputList[sub_index][self.__parsedArticleHeadersDict["Time"]]) - self.__pushDislikeTimeTranslator(outputList[index][self.__parsedArticleHeadersDict["Time"]]) <= self.__timeSlotForMerging:
                    if outputList[sub_index][self.__parsedArticleHeadersDict["User_ID"]] == outputList[index][self.__parsedArticleHeadersDict["User_ID"]]:
                        outputList[index][self.__parsedArticleHeadersDict["Main_Content"]] += str(("，" if not (self.__punctuationDict.get(outputList[index][self.__parsedArticleHeadersDict["Main_Content"]][len(outputList[index][self.__parsedArticleHeadersDict["Main_Content"]]) - 1], False) or maxWordsIndicator) else "") + outputList[sub_index][self.__parsedArticleHeadersDict["Main_Content"]])
                        maxWordsIndicator = outputList[sub_index][self.__parsedArticleHeadersDict["Max_Word_Indicator"]]
                        del outputList[sub_index]
                    else:
                        sub_index += 1
                index += 1
            else:
                del outputList[index]
        return outputList

    def saveFile(self):
        threading.Thread(target=self.__saveTitleUrlListCore(), daemon=True).run()
        threading.Thread(target=self.__saveParsedArticleListCore(), daemon=True).run()
    
    def __saveTitleUrlListCore(self):
        if not os.path.isdir(self.__filePath):
            os.mkdir(self.__filePath)
        with open(self.__filePath +"Articles_URL_List.csv", "a", encoding="utf-8") as titleUrlFile:
            titleUrlFile.write("\"" + "\", \"".join(self.__titleUrlHeadersList) + "\"\n")
            for line in self.__titleUrlList:
                titleUrlFile.write("\"" + "\", \"".join(line) + "\"\n")
    
    def __saveParsedArticleListCore(self):
        if not os.path.isdir(self.__filePath):
            os.mkdir(self.__filePath)
        for article in self.__parsedArticleList:
            subFolderName = article[0][self.__parsedArticleHeadersDict["Serial_Number"]].split("#")[0]
            articleFileName = article[0][self.__parsedArticleHeadersDict["Serial_Number"]].split("#")[1]
            if not os.path.isdir(self.__filePath + subFolderName):
                os.mkdir(self.__filePath + subFolderName)
            with open(self.__filePath + subFolderName + "\\" + articleFileName + ".csv", "w", encoding="utf-8") as parsedArticleFile:
                parsedArticleFile.write("\"" + "\", \"".join(self.__parsedArticleHeadersList) + "\"\n")
                for line in article:
                    parsedArticleFile.write("\"" + "\", \"".join(line) + "\"\n")
            parsedArticleFile.close()

    def loadFile(self):
        with open(self.__filePath + "Articles_URL_List.csv", "r", encoding="utf-8") as loadTitleUrlFile:
            self.__titleUrlList
        return

    def printDict(self):
        print(self.__boardListDict)
        print(self.__titleUrlListDict)

    def close(self):
        self.__pttConsoleDriver.close()
        self.__pttWebDriver.close()


def __unittest_main():
    ptt_user_name = input("Please enter your PTT account: ")
    ptt_user_password = input("Please enter your passwords: ")

    wished_popularity_filter = "-45"
    wished_articles_num = 250
    wished_board_list = ["NBA"]

    print("Initialize...")
    testObj = PttDataQuery(ptt_user_name, ptt_user_password, wished_board_list, wished_articles_num, wished_popularity_filter)
    print("Start to fetch URLs into a list...")
    testObj.getPttTitleUrl()

    print("Start to parse articles...")
    testObj.getPttParsedArticle()

    print("Save results...")
    testObj.saveFile()

    testObj.close()
    print("The whole procedure is completed")
    return 0


if __name__ == "__main__":
    __unittest_main()
