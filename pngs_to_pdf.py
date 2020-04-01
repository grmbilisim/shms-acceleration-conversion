
import os
from fpdf import FPDF

pdf = FPDF()

workingDir = r'/home/grm/acc-data-conversion/working/no_ui/plots/2020-01-11/uniform_plot_ranges_extended_zeropad_removed_7000-40500'


imageList = [f for f in os.listdir(workingDir) if 'bandpassed' in f]
imagePaths = [os.path.join(workingDir, f) for f in imageList]

xNum = 1
yNum = 1
for image in imagePaths:
	x = xNum * 25
	y = yNum * 25
	pdf.add_page()
	pdf.image(image, x, y, 210, 297)
	xNum += 1
	yNum += 1
pdf.output(os.path.join(workingDir, "testpdf.pdf"), "F")
