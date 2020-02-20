import os
from selenium import webdriver
from selenium.common.exceptions import SessionNotCreatedException

articleListHeadersList = ["Board_Name", "Article_URL", "Fetching_Time"]
parsedOutPutHeadersList = ["Serial_Number", "User_ID", "Main_Content", "Time", "Max_Word_Indicator"]

articleListHeadersDict = dict(zip(articleListHeadersList, range(len(articleListHeadersList))))
parsedOutPutHeadersDict = dict(zip(parsedOutPutHeadersList, range(len(parsedOutPutHeadersList))))

invert_articleListHeadersDict = {values: keys for keys, values in articleListHeadersDict.items()}
invert_parsedOutPutHeadersDict = {values: keys for keys, values in parsedOutPutHeadersDict.items()}

# Use webdriver to initiate a browser. To implement such action, we need a binary hooker to launch the browser
# To receive more details, please refer to: https://selenium-python.readthedocs.io/installation.html#introduction
# Current location: project's root
def webdriver_initiate(hooker_directory_path):
    # Firefox x86 handling
    try:
        return webdriver.Firefox(executable_path=str(hooker_directory_path + "\\Libs\\WebDriver\\geckodriver_x86.exe"))

    # Firefox x64 handling
    except SessionNotCreatedException:
        return webdriver.Firefox(executable_path=str(hooker_directory_path + "\\Libs\\WebDriver\\geckodriver_x64.exe"))

    # Google Chrome Handling
    except:
        return webdriver.Chrome(executable_path=str(hooker_directory_path + "\\Libs\\WebDriver\\chromedriver.exe"))


# Some PTT boards has an "age-limited" warning page. Our program can skip that warning page automatically by clicking the "YES" button.
def pass_ptt_age_block(driver):
    while True:
        try:    # Handling "age-limited" warning page. Use HTML's attributions to locate a specific block (in this case: submitting-button), then proceed relative actions: clicking "YES".
            warning_submitting_button = driver.find_elements_by_xpath("//button[@type='submit' and @name='yes']")[0]
            warning_submitting_button.click()
            return driver
        except (AttributeError, IndexError):  # Get target-page's source code
            return driver


def __unittest_main():
    test_driver = webdriver_initiate(os.path.abspath(""))
    test_driver.get("https://www.google.com/")
    test_driver.close()
    return


if __name__ == "__main__":
    __unittest_main()
