__author__ = 'jason'
from PIL import Image, ImageDraw, ImageFont
import itertools
import numpy as np
import click

def draw_symbol(symbol, font='impact.ttf', x=0, y=0, size=32, im_width=64, im_height=64):
    """
    Create an image of the given symbol with specified parameters
    """
    im = Image.new('RGB', (im_width, im_height), (0,0,0))
    font = ImageFont.truetype(font, size)
    draw = ImageDraw.Draw(im)
    draw.text((x,y), symbol, font=font)
    return im

def max_width_height(fonts, symbols, size):
    """
    Compute the maximum width and height of the symbols in the given fonts
    (to prevent going over the edge when drawing)
    """
    widths = []
    heights = []
    for font, symbol in itertools.product(fonts, symbols):
        font = ImageFont.truetype(font, size)
        w, h = font.getsize(symbol)
        widths.append(w)
        heights.append(h)
    return max(widths), max(heights)

def create_dataset_pytables(symbols, fonts, sizes, fname, width=64, height=64, compression=None):
    import tables

    mw, mh = max_width_height(fonts, symbols, max(sizes))
    xx = range(0, width - mw, 1)
    yy = range(0, height - mh, 1)
    combinations = list(itertools.product(symbols, fonts, sizes, xx, yy))
    n_combinations = len(combinations)
    if compression:
        filters = tables.Filters(complevel=1, complib='zlib')
    else:
        filters = None

    print("Generating {num} images".format(num=n_combinations))

    font_id = dict(zip(fonts, range(len(fonts))))

    table_handle = tables.open_file(fname, mode='w')

    images = table_handle.createEArray(table_handle.root, 'images',
                                   tables.Float32Atom(), shape=(0, width, height, 3), filters=filters, expectedrows=n_combinations)
    labels = table_handle.createEArray(table_handle.root, 'labels',
                                   tables.UInt8Atom(), shape=(0,1), expectedrows=n_combinations)
    font_label = table_handle.createEArray(table_handle.root, 'fonts',
                                   tables.UInt8Atom(), shape=(0,1), expectedrows=n_combinations)
    with click.progressbar(combinations, label="Generating {num} images".format(num=len(combinations))) as w_combinations:
        for symbol, font, size, x, y in w_combinations:
            im = draw_symbol(symbol, font, x, y, size, im_width=width, im_height=height)
            a_img = np.array(im.getdata()).reshape((1, width, height, 3))
            images.append(a_img/255.0)
            labels.append(np.uint8(symbols.find(symbol)).reshape(1,1))
            font_label.append(np.uint8(font_id[font]).reshape(1,1))
            table_handle.flush()
    print("Done")
    table_handle.close()

def create_dataset(symbols, fonts, sizes, prefix, width=64, height=64):
    """
    Create a dataset by generating images of all possible symbol/parameter combinations
    """
    mw, mh = max_width_height(fonts, symbols, max(sizes))
    xx = range(0, width - mw, 1)
    yy = range(0, height - mh, 1)
    combinations = list(itertools.product(symbols, fonts, sizes, xx, yy))
    labels = []
    with click.progressbar(combinations, label="Generating {num} images".format(num=len(combinations))) as w_combinations:
        for symbol, font, size, x, y in w_combinations:
            im = draw_symbol(symbol, font, x, y, size)
            fontname = font.split('.')[0]
            filename = './data/{prefix}-{symbol}-{x}-{y}-{font}-{size}.png'.format(prefix=prefix, font=fontname, symbol=symbol, size=size, x=x, y=y)
            labels.append((filename, symbol, fontname, size, x, y))
            with open(filename, 'w') as out:
                im.save(out, "PNG")

if __name__ == "__main__":
    # Fonts on ubuntu are in /usr/share/fonts/truetype
    # the most common ones are in /usr/share/fonts/truetype/msttcorefonts
    fonts = ['impact.ttf', 'comic.ttf', 'courbd.ttf', 'georgia.ttf', 'verdana.ttf', 'Andale_Mono.ttf']

    fname = 'digits.hdf5'
    create_dataset_pytables(symbols='abcdefghij', fonts=fonts, sizes=[10, 12, 16, 18, 20], width=32, height=32, fname=fname, compression=True)

    # To use:
    # import h5py
    # data = h5py.File('digits2.hdf5')
    # data['images'][0] would be the first image
    # data['labels'[0] would be the first label

