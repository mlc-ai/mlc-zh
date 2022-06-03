# MLC: Machine Learning Compiler

## Installation

```bash
conda env update -n mlc -f static/build.yml
conda activate mlc
pip install git+https://github.com/d2l-ai/d2l-book
```

## Building

### Building without Evaluation

Change `eval_notebook = True` to `eval_notebook = False` in `config.ini`.

## Building HTML

```bash
d2lbook build html
```

### Building PDF

```bash
# Install dependencies
sudo apt-get install texlive-full
sudo apt-get install librsvg2-bin
sudo apt-get install pandoc  # If not working, conda install pandoc

# Build PDF
d2lbook build pdf
```

### Fonts for PDF

```bash
wget https://raw.githubusercontent.com/d2l-ai/utils/master/install_fonts.sh
sudo bash install_fonts.sh
```

## Install Fonts

```bash
wget -O source-serif-pro.zip https://www.fontsquirrel.com/fonts/download/source-serif-pro
unzip source-serif-pro -d source-serif-pro
sudo mv source-serif-pro /usr/share/fonts/opentype/

wget -O source-sans-pro.zip https://www.fontsquirrel.com/fonts/download/source-sans-pro
unzip source-sans-pro -d source-sans-pro
sudo mv source-sans-pro /usr/share/fonts/opentype/

wget -O source-code-pro.zip https://www.fontsquirrel.com/fonts/download/source-code-pro
unzip source-code-pro -d source-code-pro
sudo mv source-code-pro /usr/share/fonts/opentype/

wget -O Inconsolata.zip https://www.fontsquirrel.com/fonts/download/Inconsolata
unzip Inconsolata -d Inconsolata
sudo mv Inconsolata /usr/share/fonts/opentype/

sudo fc-cache -f -v
```
