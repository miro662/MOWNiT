import sys
from typing import Tuple
from dataclasses import dataclass

import numpy as np
import freetype
import string
from intervaltree import IntervalTree

from fft_analyse import load_image, analyse, cutoff

size = 32
LETTERS = string.ascii_lowercase + string.digits + ",?!/"

def load_font_patterns(characters: str, face: str = "fonts/arial.ttf"):
    face = freetype.Face(face)
    face.set_char_size(size*64)
    result = {}
    for character in characters:
        face.load_char(character)
        shape = (
            face.glyph.bitmap.rows,
            face.glyph.bitmap.width 
        )
        array_np = np.fromiter(face.glyph.bitmap.buffer, dtype=np.ubyte)
        matrix = np.reshape(array_np, shape)
        result[character] = matrix
    
    return result


def match_points(image, pattern):
    analysed = analyse(image, pattern)
    pattern_max = np.amax(analyse(pattern, pattern))

    points = cutoff(analysed, 0.8, pattern_max=pattern_max)
    return points


def deduplicate(points):
    i = 0
    while i < len(points):
        # deduplicating i-th element
        # find duplicates of i
        duplicates = []
        for j in range(i, len(points)):
            # if duplicate, add j to array
            if is_duplicate(points[i], points[j]):
                duplicates.append(j)
        
        avg_x = sum(
            p[0] for a, p in enumerate(points) 
            if a in duplicates or a == i
        ) / len(points)
        avg_y = sum(
            p[1] for a, p in enumerate(points) 
            if a in duplicates or a == i
        ) / len(points)
        new_point = (avg_x, avg_y)
        # replace duplicates with new point
        for j in duplicates:
            points[j] = points[i]     
        points = list(set(points))
        i += 1

    return set(points)

def is_duplicate(p1, p2):
    p1x, p1y, _ = p1
    p2x, p2y, _ = p2
    distance = np.sqrt((p2y - p1y) ** 2 + (p2x - p1x) ** 2)
    return distance < (size / 2)    


@dataclass
class Character:
    char: str
    strength: np.float
    coords: Tuple[np.float, np.float]


if __name__ == '__main__':
    img = load_image(sys.argv[1], inverted=True)
    patterns = load_font_patterns(LETTERS)
    # find letters and deduplicate
    found_points = {
        letter: deduplicate(match_points(img, pattern)) 
        for letter, pattern in patterns.items()
    }
    # transform into characters
    characters = []
    for letter, points in found_points.items():
        for point in points:
            characters.append(
                Character(
                    char=letter,
                    strength=point[2],
                    coords=(point[0], point[1])
                )
            )

    # divide into rows
    rows = []
    for char in characters:
        _, y = char.coords
        row_index = -1
        # find a row that we can use
        for i, (row_y, _) in enumerate(rows):
            if abs(row_y - y) < (size / 2):
                row_index = i
                break
        
        # if we can use row
        if row_index != -1:
            # use it
            l = len(rows[row_index][1])
            rows[row_index][1].append(char)
            rows[row_index] = ((rows[row_index][0] * l + y) / (l + 1), rows[row_index][1])
        # othwerwise, create new row
        else:
            rows.append((y, [char]))

    # sort rows by y-position
    rows = sorted(rows, key=lambda x: x[0])
    lines = []
    for _, row in rows:
        # sort in rows by strength
        row = sorted(row, key=lambda x: x.strength, reverse=True)    

        tree = IntervalTree()
        for ch in row:
            end = ch.coords[0] - 1
            begin = end - patterns[ch.char].shape[1] + 2  
            print(ch.char, begin, end, tree.overlap(begin, end))
            if not tree.overlap(begin, end):
                tree[begin:end] = ch
        
        row_items = sorted((x.data for x in tree.items()), key=lambda x: x.coords[0])
        line = ""
        last = row_items[0]
        for ch in row_items:
            if abs(last.coords[0] - ch.coords[0]) > 1.3 * patterns[ch.char].shape[1]:
                line += ' '
            line += ch.char
            last = ch
        
        lines.append(line)
        print()
    
    text = '\n'.join(lines)
    print("result:\n")
    print(text)
    print("\noccurences:")
    occurences = {
        letter: len(list(ch for ch in text if ch == letter))
        for letter in LETTERS
    }
    for letter, amount in occurences.items():
        if amount > 0:
            print("{}:  {}".format(letter, amount))

