## Requirements

- Python 3.6
- ImageMagick 6
- (The Wand package does not support ImageMagick 7. To install the previous version on a Mac run

    `brew install imagemagick@6`

Create a symlink to this newly installed dylib file as mentioned in other answer to get things working.

    `ln -s /usr/local/Cellar/imagemagick@6/<your specific 6 version>/lib/libMagickWand-6.Q16.dylib /usr/local/lib/libMagickWand.dylib`
)

## Installing required packages
    `pip install --no-cache-dir spacy`
    `python -m spacy download en`
    `pip3 install -r requirements.txt`

## To run

For PDFs that have been digitally generated, place all PDFs in a folder and run

    `python anonymise.py 'path/to/files'`

The character recognition will not perform well if the pages are skewed (e.g. from scanned PDFs) so they need to be corrected. Performing skewness correction on pages that do not require it, will not change them but it will increase running time (~1 min per page). To anonymise PDFs with skewness correction run

    `python anonymise.py 'path/to/files' 1`

## License

Copyright (C) 2018 Wealth Wizards, Ltd. Distributed under the terms of the [MIT License](https://opensource.org/licenses/MIT).
