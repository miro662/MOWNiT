import sys
from typing import Tuple
from dataclasses import dataclass

import numpy as np
import freetype
import string

from fft_analyse import load_image, analyse, cutoff

size = 40 

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

    points = cutoff(analysed, 0.9, pattern_max=pattern_max)
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
        
        new_point = points[i]

        # replace duplicates with new point
        for j in duplicates:
            points[j] = points[i]     
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
    patterns = load_font_patterns(
        string.ascii_lowercase + string.digits
    )
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
    for _, row in rows:
        # sort in rows by x-position
        row = sorted(row, key=lambda x: x.coords[0])    

        line = ""
        last_ch_x = row[0].coords[0]
        for ch in row:
            distance = ch.coords[0] - last_ch_x
            ch_size = patterns[ch.char].shape[0]
            if distance > ch_size * 1.1:
                line += ' ' 
            line += ch.char
            last_ch_x = ch.coords[0]
        print(line)
