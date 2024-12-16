import string
import numpy as np
import collections
import math


class NaiveBayes:
    def fit(self, x, y):
        classes, count = np.unique(y, return_counts=True)  # находим классы, и считаем сколько шаблонов каждого

        self.n = len(classes)  # сколько всего классов

        self.prior = count / len(y)  # вероятности классов

        # найдем вероятности для слов принадлежать класс
        # сколько раз каждое слово входило в каждый класс
        self.words = [collections.Counter([]) for i in range(self.n)]
        for i in range(len(y)):
            cls = y[i]
            self.words[cls] += collections.Counter(x[i])

        self.cls_words_count = sum(len(line) for line in x)  # общее количество слов

    # вероятность слова в классе
    def get_prob_word(self, word, cls):
        return (self.words[cls][word] + 1) / self.cls_words_count

    # предсказание для массива тестов
    def predict(self, x):
        classes_probs = [
            [self.prior[cls] * math.prod(self.get_prob_word(word, cls) for word in test) for cls in range(self.n)]
            for test in x]
        ans = [np.argmax(test) for test in classes_probs]
        return ans


def imporve_text(lines):
    ans = []
    for line in lines:
        ans.append(''.join(char for char in line if char not in string.punctuation).lower())
    return ans

# read NOT spam messages
with open("not spam.txt", "r", encoding="UTF-8") as f:
    lines = f.readlines()
    lines = imporve_text(lines)
    x0 = list(map(str.split, lines))
    for i in range(len(x0)):
        x0[i].remove("тема")
        x0[i].remove("содержание")
        x0[i].pop(0)
    y0 = [0 for i in range(len(lines))]

# read spam messages
with open("spam.txt", "r", encoding="UTF-8") as f:
    lines = f.readlines()
    lines = imporve_text(lines)
    x1 = list(map(str.split, lines))
    for i in range(len(x1)):
        x1[i].remove("тема")
        x1[i].remove("содержание")
        x1[i].pop(0)
    y1 = [1 for i in range(len(lines))]

# получили обучающую выборку
x1.extend(x0)
y1.extend(y0)

# тренируем классификатор!
bayes = NaiveBayes()
bayes.fit(x1, y1)

with open("test.txt", "r") as f:
    lines = f.readlines()
    lines = imporve_text(lines)
    test = list(map(str.split, lines))
    # получаем предсказания для тестов из текстового файла
    for ans in bayes.predict(test):
        print(f"Письмо принадлежит классу {'спам' if bool(ans) else 'не спам'}")
