from freetype import *
from PIL import Image, ImageOps

if __name__ == '__main__':
    import numpy
    import matplotlib.pyplot as plt

    face = Face('fonts/arial.ttf')
    text = 'lor2em ipsu3m, sit333 do5lor amet?'
    face.set_char_size( 40*64 )
    slot = face.glyph

    # First pass to compute bbox
    width, height, baseline = 0, 0, 0
    previous = 0
    for i,c in enumerate(text):
        face.load_char(c)
        bitmap = slot.bitmap
        height = max(height,
                     bitmap.rows + max(0,-(slot.bitmap_top-bitmap.rows)))
        baseline = max(baseline, max(0,-(slot.bitmap_top-bitmap.rows)))
        kerning = face.get_kerning(previous, c)
        width += (slot.advance.x >> 6) + (kerning.x >> 6)
        previous = c

    Z = numpy.zeros((height,width), dtype=numpy.ubyte)

    # Second pass for actual rendering
    x, y = 0, 0
    previous = 0
    for c in text:
        face.load_char(c)
        bitmap = slot.bitmap
        top = slot.bitmap_top
        left = slot.bitmap_left
        w,h = bitmap.width, bitmap.rows
        y = height-baseline-top
        kerning = face.get_kerning(previous, c)
        x += (kerning.x >> 6)
        Z[y:y+h,x:x+w] += numpy.array(bitmap.buffer, dtype='ubyte').reshape(h,w)
        x += (slot.advance.x >> 6)
        previous = c
    
    im = Image.fromarray(Z)
    im = ImageOps.invert(im)
    im.save("samples/text.png")