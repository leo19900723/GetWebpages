import random
import copy
import math
import turtle
import difflib
import collections
import concurrent.futures
import threading
import time
import socket


class ListNode:
    def __init__(self, val, nextNode=None):
        self.val = val
        self.next = nextNode

    @classmethod
    def fromArray(cls, array):
        if array:
            return cls(array[0], cls.fromArray(array[1:]))
        else:
            return None

    def printList(self):
        tmp = self
        while tmp:
            print(tmp.val, "--> ", end="")
            tmp = tmp.next
        print(tmp)


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def print2D(maze):
    print("<---------START--------->")
    for row in maze:
        print(row)
    print("<----------END---------->")


# ----------------------------------------------------Tools above-------------------------------------------


class Solution:

    def permute(self, nums):
        lookup = {}
        tempAns = []
        ans = []
        for index in range(len(nums)):
            self.backTrack(nums, lookup, 0, 0, index, tempAns, ans)
        return ans

    def backTrack(self, nums, lookup, startIndex, start, end, tempAns, ans):
        for index in range(startIndex, len(nums)):
            if lookup.get(nums[index], "Not Found") == "Not Found":
                tempAns.append(nums[index])
                lookup[nums[index]] = "locked"
                if start == end:
                    ans.append(tempAns.copy())
                self.backTrack(nums, lookup, index + 1, start + 1, end, tempAns, ans)
                del tempAns[-1]
                del lookup[nums[index]]

    def heapfy(self, array):
        for index in range(len(array) - 1, -1, -1):
            try:
                if array[2 * index + 2] < array[2 * index + 1]:
                    array[2 * index], array[2 * index - 1] = array[2 * index - 1], array[2 * index]
            except:
                continue
            finally:
                try:
                    if array[index] < array[2 * index]:
                        array[index], array[2 * index] = array[2 * index], array[index]
                except:
                    pass
            print(array)

    def sortList(self, head) -> ListNode:
        if head and head.next:
            runner, mid, preMid = head, head, head
            while runner and runner.next:
                runner = runner.next.next
                preMid = mid
                mid = mid.next

            preMid.next = None

            listA = self.sortList(head)
            listB = self.sortList(mid)
            return self.merge(listA, listB)
        elif head:
            return head
        else:
            return None

    def merge(self, listA, listB):
        if listA and listB is None:
            return listA
        elif listA is None and listB:
            return listB
        elif listA and listB:
            ans = listA if listA.val < listB.val else listB
            newListB = listA if listA.val >= listB.val else listB
            ans.next = self.merge(ans.next, newListB)
            return ans
        else:
            return None

    def updateBoard(self, board, click):
        print(click, "-----------")

        for row in board:
            print(row)

        if board[click[0]][click[1]] == "M":
            board[click[0]][click[1]] = "X"
        elif board[click[0]][click[1]] == "E":
            mineCounter = 0

            mineCounter += 1 if click[0] > 0 and board[click[0] - 1][click[1]] == "M" else 0
            mineCounter += 1 if click[0] > 0 and click[1] < len(board[0]) - 1 and board[click[0] - 1][
                click[1] + 1] == "M" else 0
            mineCounter += 1 if click[1] < len(board[0]) - 1 and board[click[0]][click[1] + 1] == "M" else 0
            mineCounter += 1 if click[0] < len(board) - 1 and click[1] < len(board[0]) - 1 and board[click[0] + 1][
                click[1] + 1] == "M" else 0
            mineCounter += 1 if click[0] < len(board) - 1 and board[click[0] + 1][click[1]] == "M" else 0
            mineCounter += 1 if click[0] < len(board) - 1 and click[1] > 0 and board[click[0] + 1][
                click[1] - 1] == "M" else 0
            mineCounter += 1 if click[1] > 0 and board[click[0]][click[1] - 1] == "M" else 0
            mineCounter += 1 if click[0] > 0 and click[1] > 0 and board[click[0] - 1][click[1] - 1] == "M" else 0

            if mineCounter:
                board[click[0]][click[1]] = str(mineCounter)
            else:
                board[click[0]][click[1]] = " "
                if click[0] > 0:
                    self.updateBoard(board, [click[0] - 1, click[1]])
                if click[0] > 0 and click[1] < len(board[0]) - 1:
                    self.updateBoard(board, [click[0] - 1, click[1] + 1])
                if click[1] < len(board[0]) - 1:
                    self.updateBoard(board, [click[0], click[1] + 1])
                if click[0] < len(board) - 1 and click[1] < len(board[0]) - 1:
                    self.updateBoard(board, [click[0] + 1, click[1] + 1])
                if click[0] < len(board) - 1:
                    self.updateBoard(board, [click[0] + 1, click[1]])
                if click[0] < len(board) - 1 and click[1] > 0:
                    self.updateBoard(board, [click[0] + 1, click[1] - 1])
                if click[1] > 0:
                    self.updateBoard(board, [click[0], click[1] - 1])
                if click[0] > 0 and click[1] > 0:
                    self.updateBoard(board, [click[0] - 1, click[1] - 1])

        return board

    def compute_number_score(self, number):
        ans = 4 if not number % 3 else 0
        number = str(number)
        c2PointerLeft, c2PointerRight = None, None
        powPointer, powLength = 0, 1

        for charIndex in range(len(number)):
            if number[charIndex] == "7":
                ans += 5

            if number[charIndex] == "2":
                if c2PointerLeft is None:
                    c2PointerLeft = charIndex
                else:
                    c2PointerRight = charIndex
            else:
                if c2PointerLeft and c2PointerRight:
                    ans += 6 * (c2PointerRight - c2PointerLeft)
                c2PointerLeft, c2PointerRight = None, None

            if int(number[powPointer]) - int(number[charIndex]) == 1:
                powLength += 1
            elif charIndex:
                ans += powLength ** 2
                powLength = 1

            powPointer = charIndex

            if not int(number[charIndex]) % 2:
                ans += 3

        if c2PointerLeft is not None and c2PointerRight is not None:
            ans += 6 * (c2PointerRight - c2PointerLeft)
        ans += powLength ** 2

        return ans

    def lock_use_analyzer(self, task):
        stack = []

        for eventID, event in enumerate(task):
            if event[:7] == "ACQUIRE":
                stack.append(event)
            else:
                if stack[-1][7:] != event[7:]:
                    return eventID + 1
                else:
                    stack.pop()
        return 0 if not stack else len(task) + 1

    def countSubstrings(self, s: str) -> int:
        ans = 0

        for index in range(len(s)):

            leftPointer = index - 1
            rightPointer = index + 1

            while leftPointer >= 0 and rightPointer < len(s) and s[leftPointer] == s[rightPointer]:
                ans += 1
                leftPointer -= 1
                rightPointer += 1

            if index < len(s) - 1 and s[index] == s[index + 1]:
                ans += 1
                leftPointer = index - 1
                rightPointer = index + 2
                while leftPointer >= 0 and rightPointer < len(s) and s[leftPointer] == s[rightPointer]:
                    ans += 1
                    leftPointer -= 1
                    rightPointer += 1

        return ans + len(s)

    def balancedStringSplit(self, s: str) -> int:
        stack = []
        ans = 0
        index = 0

        while index < len(s):
            print(stack)
            if s[index] == "R":
                stack.append(s[index])
                index += 1
            else:
                popLen = 0
                while index < len(s) and s[index] == "L":
                    popLen += 1
                    index += 1
                stack = stack[:-popLen]
                ans += 1

        return ans

    def sumLessThanInput(self, array, x):
        array = sorted(array)
        leftIndex, rightIndex = 0, len(array) - 1
        ans = 0

        while rightIndex > leftIndex:
            if array[rightIndex] + array[leftIndex] > x:
                rightIndex -= 1
            elif array[rightIndex] + array[leftIndex] <= x:
                ans += rightIndex - leftIndex
                leftIndex += 1

        return ans

    def ratInTheMaze(self, maze, startX, startY, endX, endY):
        ans = []

        def backTracking(tmpAns, x, y):
            nonlocal ans, maze, endX, endY
            if 0 <= x < len(maze) and 0 <= y < len(maze[-1]) and maze[x][y] and not tmpAns[x][y]:
                tmpAns[x][y] = 1
                if x == endX and y == endY:
                    ans.append(copy.deepcopy(tmpAns))
                backTracking(tmpAns, x - 1, y)
                backTracking(tmpAns, x, y + 1)
                backTracking(tmpAns, x + 1, y)
                backTracking(tmpAns, x, y - 1)

                tmpAns[x][y] = 0

        backTracking([[0 for _ in range(len(maze[row]))] for row in range(len(maze))], startX, startY)

        return ans

    def treasureIslandI(self, maze, startX, startY):
        ans = []
        ansStep = math.inf

        def backTracking(level, tmpAns, x, y):
            nonlocal ans, ansStep, maze

            if 0 <= x < len(maze) and 0 <= y < len(maze[-1]) and (maze[x][y] == "O" or maze[x][y] == "X"):
                maze[x][y] = "V" if maze[x][y] == "O" else "X"
                tmpAns.append([x, y])

                if maze[x][y] == "X" and level < ansStep:
                    ans = tmpAns.copy()
                    ansStep = level

                backTracking(level + 1, tmpAns, x - 1, y)
                backTracking(level + 1, tmpAns, x, y + 1)
                backTracking(level + 1, tmpAns, x + 1, y)
                backTracking(level + 1, tmpAns, x, y - 1)

                maze[x][y] = "O" if maze[x][y] == "V" else "X"
                del tmpAns[-1]

        backTracking(0, [], startX, startY)

        return ans

    def moviesOnFlight(self, movie_duration, d):
        movie_duration = sorted(movie_duration)
        target = d - 30
        ans = [0, 0]

        for index in range(len(movie_duration)):
            if movie_duration[index] <= target:
                startIndex = index + 1
                endIndex = len(movie_duration) - 1

                while startIndex <= endIndex:
                    midIndex = (startIndex + endIndex) // 2
                    if (movie_duration[midIndex] + movie_duration[index]) > target >= (
                            movie_duration[midIndex - 1] + movie_duration[index]) and midIndex - 1 != index:
                        if (movie_duration[midIndex - 1] + movie_duration[index]) > ans[0] + ans[1]:
                            ans = [movie_duration[midIndex - 1], movie_duration[index]]
                            break
                    elif (movie_duration[midIndex - 1] + movie_duration[index]) <= target:
                        startIndex = midIndex + 1
                    else:
                        endIndex = midIndex - 1
            else:
                break

        return ans

    def kNearestPostOffices(self, myPlace, postOffices, k):
        return sorted(postOffices, key=lambda x: abs(x[0] - myPlace[0]) ** 2 + abs(x[1] - myPlace[1]) ** 2)[:k]


class Member(object):

    # Assume True for the variable role indicated as a "Truth Teller", False indicated as "Liar".
    def __init__(self, idNum, role):
        self.__idNum = idNum
        self.__role = role

    def idNum(self):
        return self.__idNum

    def role(self):
        return self.__role

    def response(self, opponentRole):
        if self.__role:
            return opponentRole
        else:
            return random.randrange(2)


def distinguishLiar(totalMember, liarSize):
    setA, setB, index = [], [], 0
    truthTellerSet, liarSet = set(), set()

    # ask 2 * liarSize questions.
    while len(setA) + len(setB) // 2 <= liarSize:
        # the incoming member answer the question
        if not setA or totalMember[index].response(setA[-1].role()):
            # if the answers is true or set A is empty, add to the set A
            setA.append(totalMember[index])
        else:
            # if the answer is not true, move the last member of set A to set B and move the incoming member to set B
            setB.append(setA.pop())
            setB.append(totalMember[index])
        index += 1

    # ask the first member of set A to verify evey member, excepting itself. n-1 questions.
    for member in totalMember:
        if member.idNum() != setA[0].idNum():
            if setA[0].response(member.role()):
                truthTellerSet.add(member)
            else:
                liarSet.add(member)

    return truthTellerSet, liarSet


def prettyPrint(words, lineLength):
    output, dp, lineIndex = [collections.deque()], [lineLength], 0

    for index in range(len(words) + 1):

        if index < len(words) and dp[lineIndex] - len(words[index]) >= 0:
            output[lineIndex].append(words[index] + " ")
            dp[lineIndex] -= (len(words[index]) + 1)
        else:
            output[lineIndex][-1] = output[lineIndex][-1][:-1]
            dp[lineIndex] += 1

            if lineIndex > 0 and (dp[lineIndex] - len(output[lineIndex - 1][-1]) - 1) > 0:
                optionError1 = (dp[lineIndex] - len(output[lineIndex - 1][-1]) - 1) ** 2 + (
                        dp[lineIndex - 1] + len(output[lineIndex - 1][-1]) + 1) ** 2
                optionError2 = dp[lineIndex] ** 2 + dp[lineIndex - 1] ** 2

                if optionError1 < optionError2:
                    dp[lineIndex] -= (len(output[lineIndex - 1][-1]) + 1)
                    dp[lineIndex - 1] += (len(output[lineIndex - 1][-1]) + 1)
                    output[lineIndex].appendleft(output[lineIndex - 1].pop() + " ")
                    output[lineIndex - 1][-1] = output[lineIndex - 1][-1][:-1]

            if index < len(words):
                output.append(collections.deque([words[index] + " "]))
                dp.append(lineLength - len(words[index]) - 1)
                lineIndex += 1

    return output


def shuffleCheck_BF(strA, strB, strTarget):
    if not strA and not strB and not strTarget:
        return True
    elif strA and strA[0] == strTarget[0] and (not strB or strA[0] != strB[0]):
        return shuffleCheck_BF(strA[1:], strB, strTarget[1:])
    elif strB and strB[0] == strTarget[0] and (not strA or strA[0] != strB[0]):
        return shuffleCheck_BF(strA, strB[1:], strTarget[1:])
    elif strA and strB and strA[0] == strB[0] == strTarget[0]:
        return shuffleCheck_BF(strA[1:], strB, strTarget[1:]) or shuffleCheck_BF(strA, strB[1:], strTarget[1:])
    else:
        return False


def shuffleCheck_DP(strA, strB, strTarget):
    dp = [[False for _ in range(len(strA) + 1)] for _ in range(len(strB) + 1)]

    for rowIndex in range(len(strB) + 1):
        for colIndex in range(len(strA) + 1):

            if rowIndex == 0 and colIndex == 0:
                dp[rowIndex][rowIndex] = True
            elif strA[colIndex - 1] == strTarget[rowIndex + colIndex - 1] and (
                    rowIndex == 0 or strB[rowIndex - 1] != strTarget[rowIndex + colIndex - 1]):
                dp[rowIndex][colIndex] = dp[rowIndex][colIndex - 1]
            elif strB[rowIndex - 1] == strTarget[rowIndex + colIndex - 1] and (
                    colIndex == 0 or strA[colIndex - 1] != strTarget[rowIndex + colIndex - 1]):
                dp[rowIndex][colIndex] = dp[rowIndex - 1][colIndex]
            elif strA[colIndex - 1] == strB[rowIndex - 1] == strTarget[rowIndex + colIndex - 1]:
                dp[rowIndex][colIndex] = dp[rowIndex][colIndex - 1] or dp[rowIndex - 1][colIndex]

    return dp[-1][-1]


def buyingStrategy(D_demandList, C_inventoryFeePerUnit, K_purchaseFee, S_stockLimitation):
    dp = [[0 for _ in range(S_stockLimitation + 1)] for _ in range(len(D_demandList))]

    for monthIndex in range(len(D_demandList)):
        for stockIndex in range(S_stockLimitation + 1):
            if not monthIndex:
                dp[monthIndex][stockIndex] = C_inventoryFeePerUnit * stockIndex + (K_purchaseFee if D_demandList[monthIndex] else 0)
            else:
                dp[monthIndex][stockIndex] = min(dp[monthIndex - 1][D_demandList[monthIndex] + stockIndex] if D_demandList[monthIndex] + stockIndex <= S_stockLimitation else math.inf, dp[monthIndex - 1][0] + C_inventoryFeePerUnit * stockIndex + (K_purchaseFee if D_demandList[monthIndex] else 0))

    return dp[-1][0]


def merge(inputA, inputB):
    ans = []
    indexA, indexB = 0, 0

    while indexA < len(inputA) and indexB < len(inputB):
        if inputA[indexA] < inputB[indexB]:
            ans.append(inputA[indexA])
            indexA += 1
        else:
            ans.append(inputB[indexB])
            indexB += 1

    if indexA < len(inputA):  # Append remained
        ans += inputA[indexA:]
    else:
        ans += inputB[indexB:]
    return ans


def mergeSort3(input):
    if len(input) > 2:
        A = mergeSort3(input[:len(input) // 3])
        B = mergeSort3(input[len(input) // 3:2 * len(input) // 3])
        C = mergeSort3(input[2 * len(input) // 3:])

        return merge(A, merge(B, C))
    elif len(input) == 2:
        return input if input[0] < input[1] else input[::-1]
    else:
        return input


def mergeSort(input):
    if len(input) > 1:
        slice1 = mergeSort(input[:len(input) // 2])
        slice2 = mergeSort(input[len(input) // 2:])

        return merge(slice1, slice2)
    else:
        return input


def knapsackProblem(S, W):
    dp = []

    for i in range(len(S) + 1):
        for j in range(W):
            if i == 0:
                dp[i][j] = 0
            else:
                dp[i][j] = max((dp[i - 1][j - S[i].weight] + S[i].value) if j >= S[i].weight else -math.inf, dp[i - 1][j])

    return dp[-1][-1]


def partition(input, startIndex, endIndex):
    partitionElement = input[startIndex]
    position = startIndex

    for index in range(startIndex + 1, endIndex):
        if input[index] < partitionElement:
            position += 1
            input[position], input[index] = input[index], input[position]

    input[position], input[startIndex] = input[startIndex], input[position]
    return position


def randomPartition(input, startIndex, endIndex):
    position = random.randrange(startIndex, endIndex)
    input[position], input[startIndex] = input[startIndex], input[position]

    partitionElement = input[startIndex]
    position = startIndex

    for index in range(startIndex + 1, endIndex):
        if input[index] < partitionElement:
            position += 1
            input[position], input[index] = input[index], input[position]

    input[position], input[startIndex] = input[startIndex], input[position]
    return position


def select(input, targetPosition):
    startIndex = 0
    endIndex = len(input)

    while startIndex < endIndex:
        midIndex = partition(input, startIndex, endIndex)
        if midIndex == targetPosition:
            return input[targetPosition]
        elif midIndex < targetPosition:
            startIndex = midIndex + 1
        else:
            endIndex = midIndex


def quickSort(input, startIndex, endIndex):
    if startIndex < endIndex:
        slicer = randomPartition(input, startIndex, endIndex)
        quickSort(input, startIndex, slicer)
        quickSort(input, slicer + 1, endIndex)
    return input


def closestPair(pointList):
    px = sorted(pointList, key=lambda x: (x[0], x[1]))  # Sort pointList by x-coordinate
    py = sorted(pointList, key=lambda x: (x[1], x[0]))  # Sort pointList by y-coordinate

    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def closetPairRec(px, py):
        if len(px) + len(py) <= 3:  # If the number of total points <= 3, return the closest pair directly
            combinedList = px + py
            ans = []

            for i in range(len(combinedList)):
                for j in range(i + 1, len(combinedList)):
                    ans.append([combinedList[i], combinedList[j], distance(combinedList[i], combinedList[j])])

            return sorted(ans, key=lambda x: x[2])[0]
        else:
            qx = px[:len(px) / 2]  # Divide point list px in two parts, qx means left part.
            qy = py[:len(py) / 2]  # Divide point list py in two parts, qy means left part.
            rx = px[len(px) / 2:]  # The right part of px
            ry = py[len(py) / 2:]  # The right part of py

            q0, q1 = closetPairRec(qx, qy)  # Get the closest pair from q, the left part of the whole point list
            r0, r1 = closetPairRec(rx, ry)  # Get the closest pair from r, the right part of the whole point list

            delta = min(distance(q0, q1), distance(r0, r1))  # use the least distance of above pairs as delta

            #  Now consider crossed-zone pair, crossed q & r. the midpoint is qx[-1].
            #  gather every point within ± delta respecting the midpoint qx[-1]. Then, sorted by y-coordinate
            Sy = sorted([point for point in px if distance(point, qx[-1]) <= delta], key=lambda x: x[1])

            #  For each point in Sy, we assume it located at the bottom of boxes, which has width and height of delta/2.
            #  There are only 15 boxes that we need to consider. For each box, it can only contain 1 point.
            #  Find out the closest pair within ± delta
            s0, s1 = [math.inf, math.inf], [-math.inf, -math.inf]
            for i in range(len(Sy)):
                j = i
                while j <= i + 15 and j < len(Sy):
                    (s0, s1) = min((s0, s1), (Sy[i], Sy[j]), key=lambda x: distance(x[0], x[1]))
                    j += 1

            #  Either (s0, s1) or (q0, q1) or (r0, r1) should be the sole answer.
            return min((s0, s1), (q0, q1), (r0, r1), key=lambda x: distance(x[0], x[1]))

    return closetPairRec(px, py)


def floydWarshall(graph):
    dp = [[graph[i][j] for j in range(len(graph[i]))] for i in range(len(graph))]

    for intermediatePoint in range(len(graph)):
        for startPoint in range(len(graph[intermediatePoint])):
            for destinationPoint in range(len(graph[intermediatePoint])):
                dp[startPoint][destinationPoint] = min(dp[startPoint][destinationPoint], dp[startPoint][intermediatePoint] + dp[intermediatePoint][destinationPoint])

    return dp


def bellmanFord(graph, startPoint, destinationPoint):
    dp = [[math.inf if node != destinationPoint else 0 for node in range(len(graph[level]))] for level in range(len(graph))]

    for level in range(1, len(dp)):
        for point in range(len(dp[level])):
            dp[level][point] = min(dp[level - 1][point], min([dp[level - 1][newPoint] + graph[point][newPoint] for newPoint in range(len(dp[level]))]))
        print2D(dp)

    return dp[-1][startPoint]


def stringAlignment(strA, strB):
    strA = "_" + strA
    strB = "_" + strB

    dp = [[0 for _ in range(len(strA))] for _ in range(len(strB))]
    alpha = 2
    delta = 2

    for charAIndex in range(len(strA)):
        for charBIndex in range(len(strB)):
            if not charAIndex and not charBIndex:
                dp[charAIndex][charBIndex] = 0
            elif not charAIndex:
                dp[charAIndex][charBIndex] = dp[charAIndex][charBIndex - 1] + delta
            elif not charBIndex:
                dp[charAIndex][charBIndex] = dp[charAIndex - 1][charBIndex] + delta
            else:
                dp[charAIndex][charBIndex] = min(dp[charAIndex - 1][charBIndex - 1] + (alpha if strA[charAIndex] != strB[charBIndex] else 0), dp[charAIndex - 1][charBIndex] + delta, dp[charAIndex][charBIndex - 1] + delta)

    return dp

def main():
    strA = list("aabcc")
    strB = list("dbbca")
    strTarget = list("aadbbbaccc")

    prettyWords = ["call", "me", "ishmael.", "some", "years", "ago,", "never", "mind", "how", "long", "precisely,",
                   "having", "little", "or", "no", "money", "in", "my", "purse,", "and", "nothing", "particular", "to",
                   "interest", "me", "on", "shore,", "I", "thought", "I", "would", "sail", "about", "a", "little",
                   "and", "see", "the", "watery", "part", "of", "the", "world."]

    d = [4, 10, 6, 3, 8, 5, 2, 1]
    c = 1
    k = 5
    s = 10

    graphFloydWarshall = [
        [0, 8, 3, 1, math.inf],
        [8, 0, 4, math.inf, 2],
        [3, 4, 0, 1, 1],
        [1, math.inf, 1, 0, 8],
        [math.inf, 2, 1, 8, 0]
    ]

    graphBellmanFord = [
        [0, math.inf, math.inf, math.inf, math.inf, math.inf],
        [-3, 0, -4, math.inf, math.inf, math.inf],
        [math.inf, math.inf, 0, math.inf, -1, -2],
        [3, math.inf, 8, 0, math.inf, math.inf],
        [4, 6, math.inf, math.inf, 0, math.inf],
        [2, math.inf, math.inf, -3, math.inf, 0]
    ]

    teacherSampleGraph = [
        [0, 3, math.inf, math.inf],
        [math.inf, 0, 12, 5],
        [math.inf, math.inf, 0, -1],
        [2, -4, math.inf, 0]
    ]

    alignmentStrA = "name"
    alignmentStrB = "mean"

    print2D(floydWarshall(teacherSampleGraph))
    print(mergeSort3(d))
    print(bellmanFord(graphBellmanFord, 1, 0))
    print2D(stringAlignment(alignmentStrA, alignmentStrB))
    print(quickSort(d, 0, len(d)))
    print(buyingStrategy(d, c, k, s))
    print(shuffleCheck_BF(strA, strB, strTarget))
    print(shuffleCheck_DP(strA, strB, strTarget))
    print2D(prettyPrint(prettyWords, 38))


if __name__ == "__main__":
    main()
