
class CubatureIntegrator:
    def __init__(self, rule, domain):
        self.rule = rule
        self.domain = domain

    def integrate(self, func):
        points, weights = self.rule.generate(self.domain)
        return sum(w * func(p) for p, w in zip(points, weights))

