import sys
import os
from distutils.util import strtobool
from PyQt5 import QtWidgets, QtCore


class StyledCheckBox(object):

    def __init__(self):
        self.__checkBoxWidget = QtWidgets.QWidget()
        self.__checkBox = QtWidgets.QCheckBox()
        self.__checkBoxLayout = QtWidgets.QHBoxLayout(self.__checkBoxWidget)
        self.__checkBoxLayout.addWidget(self.__checkBox)
        self.__checkBoxLayout.setAlignment(QtCore.Qt.AlignCenter)
        self.__checkBoxLayout.setContentsMargins(0, 0, 0, 0)

    def getCheckboxWidget(self):
        return self.__checkBoxWidget

    def getCheckbox(self):
        return self.__checkBox


class NotingUserInterface(QtWidgets.QMainWindow):

    def __init__(self):
        super(NotingUserInterface, self).__init__()
        self.__windowStartPositionX = 150
        self.__windowStartPositionY = 100
        self.__windowHeight = 600
        self.__windowWidth = 1000
        self.__inputDataList = []
        self.__outputDataList = []
        self.__checkBoxList = []
        self.__progressBar = QtWidgets.QProgressBar(self)
        self.__modifiedFlag = False
        self.__inputFileName = ""
        self.__outputFilename = ""
        self.__programTitle = "文字標記使用介面"
        self.__supCategoriesTableHeaderList = ["第三者角度", "當事人角度"]
        self.__subCategoriesTableHeaderList = ["性別", "身心", "政治", "國族", "辱罵", "其他"]

        self.__outputDataHeaderDict = {}
        commentTableHeader = []
        for supCate in self.__supCategoriesTableHeaderList:
            for subCate in self.__subCategoriesTableHeaderList:
                self.__outputDataHeaderDict[supCate + "#" + subCate] = len(self.__outputDataHeaderDict)
                commentTableHeader.append(subCate)
        for tail in ["暗示", "疑問", "Serial_Number", "Main_Content"]:
            self.__outputDataHeaderDict[tail] = len(self.__outputDataHeaderDict)
            commentTableHeader.append(tail if tail != "Main_Content" else "推文內容")

        # Create Main Frame
        self.__centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.__centralWidget)
        self.setGeometry(self.__windowStartPositionX, self.__windowStartPositionY, self.__windowWidth, self.__windowHeight)
        self.setWindowTitle(self.__programTitle)

        # Create Article Frame
        self.__articleFrame = QtWidgets.QTextBrowser(self)
        self.__articleFrame.setMaximumHeight(230)

        # Create Table Frame
        self.__tableFrame = QtWidgets.QVBoxLayout()

        # Create Table
        self.__commentTable = QtWidgets.QTableWidget(0, len(self.__outputDataHeaderDict))
        self.__commentTable.setHorizontalHeaderLabels(commentTableHeader)
        for columnHeaderIndex in range(0, len(self.__outputDataHeaderDict)):
            self.__commentTable.horizontalHeader().setSectionResizeMode(columnHeaderIndex, QtWidgets.QHeaderView.ResizeToContents)
        self.__commentTable.horizontalHeader().setSectionResizeMode(self.__outputDataHeaderDict["Main_Content"], QtWidgets.QHeaderView.Stretch)
        self.__commentTable.setColumnHidden(self.__outputDataHeaderDict["Serial_Number"], True)

        # Create Super Table Header
        self.__commentTableSupHeader = QtWidgets.QTableWidget(0, len(self.__supCategoriesTableHeaderList) + 1)
        self.__commentTableSupHeader.setHorizontalHeaderLabels(self.__supCategoriesTableHeaderList + ["請選擇檔案"])
        self.__commentTableSupHeader.horizontalHeader().setSectionResizeMode(len(self.__supCategoriesTableHeaderList), QtWidgets.QHeaderView.Stretch)
        self.__supTableHeaderResize()
        self.__commentTableSupHeader.setMaximumHeight(self.__commentTableSupHeader.horizontalHeader().height())

        self.__tableFrame.addWidget(self.__commentTableSupHeader)
        self.__tableFrame.addWidget(self.__commentTable)
        self.__tableFrame.setSpacing(0)

        # Create Load File Button
        self.__loadButton = QtWidgets.QPushButton("讀取檔案", self)
        self.__loadButton.clicked.connect(self.loadBtnHandler)

        # Create Save File Button
        self.__saveButton = QtWidgets.QPushButton("儲存檔案", self)
        self.__saveButton.clicked.connect(self.saveBtnHandler)

        # Manage the layout
        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(self.__articleFrame, 0, 0, 1, 3)
        mainLayout.addLayout(self.__tableFrame, 1, 0, 1, 3)
        mainLayout.addWidget(self.__loadButton, 2, 0)
        mainLayout.addWidget(self.__saveButton, 2, 1)
        mainLayout.addWidget(self.__progressBar, 2, 2)

        self.__centralWidget.setLayout(mainLayout)
        self.show()

    def __supTableHeaderResize(self):
        for supIndex, supCate in enumerate(self.__supCategoriesTableHeaderList):
            width = 0
            for subIndex, subCate in enumerate(self.__subCategoriesTableHeaderList):
                width += self.__commentTable.columnWidth(subIndex)
            self.__commentTableSupHeader.setColumnWidth(supIndex, width)

    def checkBoxHandler(self, index):
        self.__modifiedFlag = True
        index = index.split()
        print(self.__outputDataList[int(index[0])]["Serial_Number"], ": [", index[1], "] -->", self.__checkBoxList[int(index[0])][index[1]].getCheckbox().isChecked())
        self.__outputDataList[int(index[0])][index[1]] = str(self.__checkBoxList[int(index[0])][index[1]].getCheckbox().isChecked())

    def resetContent(self):
        self.__modifiedFlag = False
        self.__articleFrame.setPlainText("")
        while self.__commentTable.rowCount() > 0:
            self.__commentTable.removeRow(0)
        del self.__inputDataList[:]
        del self.__outputDataList[:]
        del self.__checkBoxList[:]

    def loadBtnHandler(self):
        if self.__modifiedFlag:
            reply = QtWidgets.QMessageBox.question(self, self.__programTitle, "有尚未儲存的變更，是否儲存？", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.__saveCore()

        self.__inputFileName, valid = QtWidgets.QFileDialog.getOpenFileNames(self, "", "", "CSV Files (*.csv);;All Files (*)")
        if not valid:
            return
        self.__outputFilename = str(self.__inputFileName[0]).replace(".csv", ".notedat")

        self.resetContent()
        self.__commentTableSupHeader.setHorizontalHeaderLabels(self.__supCategoriesTableHeaderList + ["現正檢視：" + self.__inputFileName[0].split("/")[-1]])
        self.__commentTableSupHeader.setDisabled(True)
        self.__commentTableSupHeader.setRowCount(1)
        self.__commentTable.setDisabled(True)
        checkBoxActionMapper = QtCore.QSignalMapper(self)

        try:
            with open(self.__outputFilename, "r", encoding="utf-8") as savedFile:
                saveFileheaderDict = dict((key, value) for value, key in enumerate(savedFile.readline().strip().strip("\"").split("\", \"")))
                savedFileString = savedFile.read()
        except FileNotFoundError:
            saveFileheaderDict = {}
            savedFileString = False

        verticalHeaderLabel = []
        with open(str(self.__inputFileName[0]), "r", encoding="utf-8") as sourceFile:
            sourceFileLines = sourceFile.read().splitlines()
            loadFileHeaderDict = dict((key, value) for value, key in enumerate(sourceFileLines[0].strip().strip("\"").split("\", \"")))
            self.__progressBar.setMaximum(int(sourceFileLines[-1].split("\", \"")[loadFileHeaderDict["Serial_Number"]].split("#")[-1]))
            for rowIndex, rowElement in enumerate(sourceFileLines[1:]):
                tempString = {}
                tempCheckBoxDict = {}
                tempOutputDataDict = {}
                verticalHeaderLabel.append(str(rowIndex))

                for key in loadFileHeaderDict.keys():
                    tempString[key] = rowElement.split("\", \"")[loadFileHeaderDict[key]].strip().strip("\"")
                self.__inputDataList.append(tempString)

                self.__commentTable.setRowCount(rowIndex + 1)
                self.__commentTable.setItem(rowIndex, self.__outputDataHeaderDict["Main_Content"], QtWidgets.QTableWidgetItem(tempString["Main_Content"]))

                for colElement in self.__outputDataHeaderDict.keys():
                    try:
                        tempOutputDataDict[colElement] = tempString[colElement]
                        tempCheckBoxDict[colElement] = None
                    except KeyError:
                        if (not savedFileString) or saveFileheaderDict.get(colElement, "Not Found") == "Not Found":
                            tempOutputDataDict[colElement] = "False"
                        else:
                            tempOutputDataDict[colElement] = savedFileString.split("\n")[rowIndex].strip("\"").split("\", \"")[saveFileheaderDict[colElement]]
                        tempCheckBoxDict[colElement] = StyledCheckBox()
                        if colElement.find(self.__supCategoriesTableHeaderList[1]) == -1:
                            tempCheckBoxDict[colElement].getCheckboxWidget().setStyleSheet("background-color: rgb(230, 255, 243);")
                        checkBoxActionMapper.setMapping(tempCheckBoxDict[colElement].getCheckbox(), str(rowIndex) + " " + colElement)
                        tempCheckBoxDict[colElement].getCheckbox().setChecked(strtobool(tempOutputDataDict[colElement]))
                        tempCheckBoxDict[colElement].getCheckbox().clicked.connect(checkBoxActionMapper.map)
                        self.__commentTable.setCellWidget(rowIndex, self.__outputDataHeaderDict[colElement], tempCheckBoxDict[colElement].getCheckboxWidget())

                self.__checkBoxList.append(tempCheckBoxDict)
                self.__outputDataList.append(tempOutputDataDict)
                self.__progressBar.setValue(rowIndex)
        sourceFile.close()

        checkBoxActionMapper.mapped["QString"].connect(self.checkBoxHandler)
        self.__commentTableSupHeader.setVerticalHeaderLabels([verticalHeaderLabel[-1]])

        self.__commentTable.resizeRowsToContents()
        self.__commentTable.setRowHidden(0, True)
        self.__commentTable.setVerticalHeaderLabels(verticalHeaderLabel)

        self.__articleFrame.setPlainText(self.__inputDataList[0]["Main_Content"].replace("<br>", "\n"))
        self.__progressBar.reset()

        self.__commentTableSupHeader.setDisabled(False)
        self.__commentTable.setDisabled(False)

    def saveBtnHandler(self):
        if self.__modifiedFlag:
            reply = QtWidgets.QMessageBox.question(self, self.__programTitle, str("是否確定要儲存？" + ("" if not os.path.isfile(self.__outputFilename) else "結果將會覆蓋原先檔案")), QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.__saveCore()
        elif len(self.__inputDataList) == 0:
            QtWidgets.QMessageBox.information(self, self.__programTitle, "請先讀取檔案", QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.information(self, self.__programTitle, "結果並無變更，無須儲存", QtWidgets.QMessageBox.Ok)

    def __saveCore(self):
        with open(self.__outputFilename, "w", encoding="utf-8") as saveFile:
            saveFile.write(str("\"" + "\", \"".join(self.__outputDataHeaderDict.keys()) + "\"\n"))
            self.__progressBar.setMaximum(len(self.__outputDataList) - 1)
            for rowIndex, element in enumerate(self.__outputDataList):
                saveFile.write(str("\"" + "\", \"".join(element.values()) + "\"\n"))
                self.__progressBar.setValue(rowIndex)
        saveFile.close()
        self.__modifiedFlag = False
        self.__progressBar.reset()
        QtWidgets.QMessageBox.information(self, self.__programTitle, "檔案儲存成功！", QtWidgets.QMessageBox.Ok)

    def __formatTableHeaderString(self, headerStringList, fixedCharactersNumber):
        outputList = []
        for element in headerStringList:
            tempOutput = "\n".join([element[index:index + fixedCharactersNumber] for index in range(0, len(element), fixedCharactersNumber)])
            outputList.append(tempOutput)
        return outputList

    def closeEvent(self, event):
        if self.__modifiedFlag:
            reply = QtWidgets.QMessageBox.question(self, self.__programTitle, "有尚未儲存的變更，是否儲存？", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
            if reply != QtWidgets.QMessageBox.Cancel:
                if reply == QtWidgets.QMessageBox.Yes:
                    self.__saveCore()
                event.accept()
            else:
                event.ignore()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = NotingUserInterface()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
