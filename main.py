import numpy as np
import collections

class NaiveBayes:
    def fit(self, x, y):
        classes, count = np.unique(y, return_counts=True) # находим классы, и считаем сколько шаблонов каждого

        n = len(classes) # сколько всего классов

        self.prior = count / len(y) # вероятности классов

        # найдем вероятности для слов принадлежать классу

