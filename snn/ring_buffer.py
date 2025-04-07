"""
Module for ring buffer implementation.
"""

import numpy as np


class RingBuffer:
    """ Implements a partially-full circular buffer. """

    def __init__(self, buffsize: int):
        self.buffsize = buffsize
        self.data = np.full(buffsize, None, dtype=object)  # Explicitly initialize with None
        self.currpos = 0  # Position where the next element should be added
        self.is_full = False  # Flag to indicate if the buffer is full

    def add(self, value: object) -> None:
        """ Adds an element at the end of the buffer. """
        self.data[self.currpos] = value
        self.currpos = (self.currpos + 1) % self.buffsize
        if self.currpos == 0:
            self.is_full = True

    def get(self, n: int = None) -> list:
        """
        Returns elements from the buffer.
        
        - No arguments: Returns the full buffer as a list.
        - n (int): Returns the most recent `n` elements (latest to oldest).
        """
        if self.currpos == 0 and not self.is_full:
            return []  # If the buffer is empty, return an empty list

        buffer_data = (
            np.concatenate((self.data[self.currpos:], self.data[:self.currpos]))
            if self.is_full else self.data[:self.currpos]
        )

        buffer_list = list(buffer_data)  # Ensure output is always a list

        if n is None:
            return buffer_list  # Return full buffer
        return buffer_list[-min(n, len(buffer_list)):]  # Return the latest `n` elements

    def length(self) -> int:
        """ Returns the current number of elements in the buffer. """
        return self.buffsize if self.is_full else self.currpos

    def is_empty(self) -> bool:
        """ Returns True if the buffer is empty, False otherwise. """
        return self.currpos == 0 and not self.is_full

    def clear(self) -> None:
        """ Resets the buffer, removing all stored elements. """
        self.data.fill(None)
        self.currpos = 0
        self.is_full = False
