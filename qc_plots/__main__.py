from argparse import ArgumentParser
from contextlib import ExitStack
from pathlib import Path
from PIL import Image
from PyPDF2 import PdfFileWriter, PdfFileReader

import tomli

from . import qc_plots2
from .constants import DEFAULT_CONFIG, DEFAULT_LIMITS


def images_to_pdf(fig_path_list, pdf_path: Path, size='medium', quality='high'):
    size_divisor = {'small': 3, 'medium': 2, 'large': 1}[size]
    quality = {'low': 30, 'medium': 60, 'high': 95}[quality]
    temp_pdf_path = pdf_path.with_suffix('.tmp.pdf')

    with ExitStack() as img_stack:
        im_list = [img_stack.enter_context(Image.open(fig_path).convert('RGB')) for fig_path in fig_path_list]
        im_list = [im.resize((im.width // size_divisor, im.height // size_divisor)) for im in im_list]
        im_list[0].save(temp_pdf_path, "PDF", quality=quality, save_all=True, append_images=im_list[1:])

        # add bookmarks to the temporary pdf file and save the final file
        pdf_in = img_stack.enter_context(open(temp_pdf_path, 'rb'))
        pdf_out = img_stack.enter_context(open(pdf_path, 'wb'))

        input = PdfFileReader(pdf_in)
        output = PdfFileWriter()

        for i, fig_path in enumerate(fig_path_list):
            output.addPage(input.getPage(i))
            output.addBookmark(Path(fig_path).stem, i)

        output.write(pdf_out)
        temp_pdf_path.unlink()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('nc_in',help='full path to the netCDF file from which plots should be made, or a path to a directory (plots will be made from all *.nc files in that directory)')
    parser.add_argument('-r','--ref',default='',help='full path to another netCDF file to use as reference')
    parser.add_argument('-c', '--context', default='', help='full path to another netCDF file to use as context (i.e. full record for the site being plotted)')
    parser.add_argument('-d', '--output-dir', default='.', help='Directory to output the .pdf plots to, the default is "outputs" in the code repo')
    parser.add_argument('--flag0',action='store_true',help='only plot flag=0 data with axis ranges only within the vmin/vmax values of each variable')
    parser.add_argument('--cmap',default='PuBu',help='valid name of a matplotlib colormap https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='full path to the input .toml file for variables to plot')
    parser.add_argument('--json-limits', default=DEFAULT_LIMITS, help='full path to the input json file for axis ranges')
    parser.add_argument('--show-all',action='store_true',help='if given, the axis ranges of the plots will automatically fit in all the data, even huge outliers')
    parser.add_argument('--roll-window',type=int,default=500,help='Size of the rolling window in number of spectra')
    parser.add_argument('--roll-gaps',default='20000 days',help='Minimum time interval for which the data will be split for rolling stats')
    parser.add_argument('--size',choices=('small','medium','large'),default='medium',help='Size of the figures, default is %(default)s')
    parser.add_argument('--quality',choices=('low','medium','high'),default='high',help='Quality of the figures, default is %(default)s')

    return vars(parser.parse_args())


def driver(nc_in, config, flag0=False, show_all=False, output_dir='.', size='medium', quality='high', **kwargs):
    with open(config) as f:
        config = tomli.load(f)

    import pdb; pdb.set_trace()
    with ExitStack() as stack:
        primary_styles = config.get('style', dict()).get('main', dict())
        data = [stack.enter_context(qc_plots2.TcconData(qc_plots2.DataCategory.PRIMARY, nc_in, styles=primary_styles))]
        plots = qc_plots2.setup_plots(config)
        fig_paths = []
        n = len(plots)
        for i, plot in enumerate(plots, start=1):
            print(f'  - Plot {i}/{n}: {plot.get_save_name()}', end='')
            try:
                this_path = plot.make_plot(data, flag0_only=flag0, show_all=show_all)
            except IndexError as err:
                print(f' SKIPPED ({err})')
            else:
                print(' DONE')
                fig_paths.append(this_path)

        pdf_name = Path(nc_in).with_suffix('.pdf' if not flag0 else '.flag0.pdf').name
        pdf_path = Path(output_dir) / pdf_name
        images_to_pdf(fig_paths, pdf_path, size=size, quality=quality)


def main():
    clargs = parse_args()
    driver(**clargs)


if __name__ == '__main__':
    main()
