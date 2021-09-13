from argparse import ArgumentParser
from contextlib import ExitStack
from pathlib import Path
import tempfile
from PIL import Image
from PyPDF2 import PdfFileWriter, PdfFileReader
import sys
import tempfile
import tomli

from . import qc_plots2, qc_email
from .constants import DEFAULT_CONFIG, DEFAULT_IMG_DIR, DEFAULT_LIMITS


def images_to_pdf(fig_path_list, pdf_path: Path, size='medium', quality='high'):
    size_divisor = {'small': 3, 'medium': 2, 'large': 1}[size]
    quality = {'low': 30, 'medium': 60, 'high': 95}[quality]
    temp_pdf_path = pdf_path.with_suffix('.tmp.pdf')

    with ExitStack() as img_stack:
        im_list = []
        for fig_path in fig_path_list:
            try:
                im_list.append(img_stack.enter_context(Image.open(fig_path).convert('RGB')))
            except AttributeError:
                # We get an attribute error if opening a file produces a None, since it can't seek
                raise IOError('Error opening plot file "{}" for concatenation into the PDF'.format(fig_path))
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


def send_email(pdf_path, nc_file, emails=(None, None), email_config=None):
    if any(e is not None for e in emails):
        if not all(e is not None for e in emails):
            print('WARNING: provided either a sender or receiver email but not both; will not use the provided email')
        else:
            print(f'Sending {pdf_path} by email from {emails[0]} to {emails[1]}')
            subject = 'TCCON QC plots: {}'.format(Path(pdf_path).name)
            body = "This email was sent from the python program qc_plots"
            qc_email.send_email(
                subject=subject,
                body=body,
                send_from=emails[0],
                send_to=emails[1],
                attachment=pdf_path
            )
            print('Email sent.')

    if email_config is not None:
        print(f'Sending {pdf_path} by email based on config file {email_config}')
        nc_basename = Path(nc_file).name
        site_id = nc_basename[:2]
        qc_email.send_email_from_config(
            email_config,
            site_id=site_id,
            attachment=pdf_path,
            nc_file=nc_basename
        )
        print('Email sent.')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('nc_in', help='full path to the netCDF file from which plots should be made, or a path to a '
                                      'directory (plots will be made from all *.nc files in that directory)')
    parser.add_argument('-r', '--ref', help='full path to another netCDF file to use as reference')
    parser.add_argument('-c', '--context', help='full path to another netCDF file to use as context (i.e. full record '
                                                'for the site being plotted)')
    parser.add_argument('-d', '--output-dir', default='.', help='Directory to output the .pdf plots to, the default is '
                                                                '"outputs" in the code repo')
    parser.add_argument('--flag0', action='store_true', help='only plot flag=0 data with axis ranges only within the '
                                                             'vmin/vmax values of each variable')
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        help='full path to the input .toml file for variables to plot')
    parser.add_argument('--limits', default=DEFAULT_LIMITS,
                        help='full path to the input .toml file for axis ranges')
    parser.add_argument('--show-all', action='store_true',
                        help='if given, the axis ranges of the plots will automatically fit in all the data, even '
                             'huge outliers')
    parser.add_argument('--use-tmp-img-dir', action='store_true', help='Save intermediate images to a temporary directory, '
                                                                       'instead of `outputs` in the code directory')
    parser.add_argument('--size', choices=('small', 'medium', 'large'), default='medium',
                        help='Size of the figures, default is %(default)s')
    parser.add_argument('--quality', choices=('low', 'medium', 'high'), default='high',
                        help='Quality of the figures, default is %(default)s')

    email_grp = parser.add_mutually_exclusive_group()
    email_grp.add_argument('--email', nargs=2, default=[None, None],
                           help='sender email followed by receiver email; by default uses the local email server, but '
                                'can also work with Outlook accounts and Gmail accounts that enabled less secured apps '
                                'access. For multiple recipients the second argument should be comma-separated email '
                                'addresses.')
    email_grp.add_argument('--email-config', help='A .toml file that sets how emails should be sent.')
    email_grp.add_argument('--gen-email-config', action='store_true',
                           help='Create a default email config TOML file and exit. The usual positional argument '
                                '(NC_IN) will be used as the path to create the TOML file at.')

    parser.add_argument('--pdb', action='store_true', help='Launch python debugger')

    return vars(parser.parse_args())


def driver(nc_in, config, limits, ref=None, context=None, flag0=False, show_all=False, output_dir='.',
           use_tmp_img_dir=False, size='medium', quality='high', emails=(None, None), email_config=None, **_):

    print(f'Using {config} as plots configuration file')
    print(f'Using {limits} as plots limits file')

    with open(config) as f:
        config = tomli.load(f)

    with ExitStack() as stack:
        if use_tmp_img_dir:
            img_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        else:
            img_dir = Path(DEFAULT_IMG_DIR)
        print(f'Will save intermediate images to {img_dir}')
            
        primary_styles = config.get('style', dict()).get('main', dict())
        data = [stack.enter_context(
            qc_plots2.TcconData(
                qc_plots2.DataCategory.PRIMARY,
                nc_in,
                styles=primary_styles
            ))]

        if context is not None:
            context_styles = config.get('style', dict()).get('context', dict())
            context_data = qc_plots2.TcconData(
                qc_plots2.DataCategory.CONTEXT,
                context,
                exclude_times=data[0].times,
                styles=context_styles
            )
            data.append(stack.enter_context(context_data))

        if ref is not None:
            ref_styles = config.get('style', dict()).get('ref', dict())
            ref_data = qc_plots2.TcconData(
                qc_plots2.DataCategory.REFERENCE,
                ref,
                styles=ref_styles,
                allowed_flag_categories={qc_plots2.FlagCategory.FLAG0}
            )
            data.append(stack.enter_context(ref_data))

        plots = qc_plots2.setup_plots(config, limits_file=limits)
        fig_paths = []
        n = len(plots)
        for i, plot in enumerate(plots, start=1):
            sys.stdout.write(f'  - Plot {i}/{n}: {plot.get_save_name()}')
            sys.stdout.flush()
            try:
                this_path = plot.make_plot(data, flag0_only=flag0, show_all=show_all, img_path=img_dir)
            except IndexError as err:
                print(f' SKIPPED ({err})')
            else:
                print(' DONE')
                fig_paths.append(this_path)

        pdf_name = Path(nc_in).with_suffix('.pdf' if not flag0 else '.flag0.pdf').name
        pdf_path = Path(output_dir) / pdf_name
        images_to_pdf(fig_paths, pdf_path, size=size, quality=quality)

    send_email(pdf_path=pdf_path, nc_file=nc_in, emails=emails, email_config=email_config)


def main():
    clargs = parse_args()
    if clargs.pop('pdb'):
        import pdb
        pdb.set_trace()

    if clargs.pop('gen_email_config'):
        qc_email.write_default_config(clargs['nc_in'])
    else:
        driver(**clargs)


if __name__ == '__main__':
    main()
