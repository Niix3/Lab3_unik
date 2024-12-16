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

        self.cls_words_count = [sum(self.words[cls].values()) for cls in
                                range(len(self.words))]  # общее количество слов

    # вероятность слова в классе
    def get_prob_word(self, word, cls):
        return (self.words[cls][word] + 1) / self.cls_words_count[cls]

    # предсказание для массива тестов
    def predict(self, x):
        classes_probs = [
                [self.prior[cls] * math.prod(self.get_prob_word(word, cls) for word in test) for cls in range(self.n)]
            for test in x]
        ans = [np.argmax(test) for test in classes_probs]
        return ans


# read NOT spam messages
with open("not spam.txt", "r", encoding="UTF-8") as f:
    lines = f.readlines()
    x0 = list(map(str.split, lines))
    for i in range(len(x0)):
        x0[i].remove("**Тема**:")
        x0[i].remove("**Содержание**:")
        x0[i].pop(0)
    y0 = [0 for i in range(len(lines))]

# read spam messages
with open("spam.txt", "r", encoding="UTF-8") as f:
    lines = f.readlines()
    x1 = list(map(str.split, lines))
    for i in range(len(x1)):
        x1[i].remove("**Тема**:")
        x1[i].remove("**Содержание**:")
        x1[i].pop(0)
    y1 = [1 for i in range(len(lines))]
# получили обучающую выборку
x1.extend(x0)
y1.extend(y0)

bayes = NaiveBayes()
bayes.fit(x1, y1)

with open("test.txt", "r") as f:
    lines = f.readlines()
    test = list(map(str.split, lines))
    for ans in bayes.predict(test):
        print(f"Письмо принадлежит классу {'спам' if bool(ans) else 'Не спам'}")
