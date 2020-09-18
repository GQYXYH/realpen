"""
Wrapper to get an image from the environment and not a state.

@Author: Moritz Schneider
"""
import cv2

from gym import ObservationWrapper, spaces


class ImageObservationWrapper(ObservationWrapper):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides
    """
    def __init__(self, env, out_shape=None):
        super(ImageObservationWrapper, self).__init__(env)
        dummy_obs = env.render("rgb_array")
        # Update observation space
        self.out_shape = out_shape

        obs_shape = out_shape if out_shape is not None else dummy_obs.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=dummy_obs.dtype)

    def observation(self, observation):
        img = self.env.render("rgb_array")
        if self.out_shape is not None:
            img = cv2.resize(img, (self.out_shape[0], self.out_shape[1]), interpolation=cv2.INTER_AREA)
        # TODO: Resulting state as information?
        return img
