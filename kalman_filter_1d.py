import numpy as np
# adapted from https://github.com/zziz/kalman-filter and https://gist.github.com/manicai/922976

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def predict_future(self, n_steps,  u = 0):
        matrix = self.F.copy()
        for _ in range(n_steps-1):
            matrix = np.dot(matrix, self.F)
        x = np.dot(matrix, self.x) + np.dot(self.B, u)
        return x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
            (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

class RealWorld:
    def __init__(self):
        self.position = 0.0
        self.velocity = 0.5
        self.time_step = 0.1
        self.time = 0.0
        self.measure = None

        # Noise on the measurements
        self.measurement_variance = 0.2

        # If we want to kink the profile.
        self.change_after = 50
        self.changed_velocity = -0.5

    def measurement(self):
        if self.measure == None:
            self.measure = (self.position
                            + np.random.normal(0, self.measurement_variance))
        return self.measure

    def step(self):
        self.time += self.time_step
        self.position += self.velocity * self.time_step

        if self.time >= self.change_after:
            self.velocity = self.changed_velocity
        self.measure = None

def example_1D():
    world = RealWorld()
    dt = world.time_step
    
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    x = np.linspace(-10, 10, 100)
    measurements = - (x**2 + 2*x - 2)  + np.random.normal(0, 2, 100)

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
    measurements = []
    predictions = []
    predictions_future = []

    for _ in range(1000):
        world.step()
        measurement = world.measurement()
        measurements.append(measurement)

        predictions.append(np.dot(H,  kf.predict())[0])
        predictions_future.append(np.dot(H,  kf.predict_future(5))[0])
        kf.update(measurement)


    import matplotlib.pyplot as plt
    plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
    plt.plot(range(len(predictions_future)), np.array(predictions_future), label = 'Kalman Filter Future Prediction')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    example_1D()
