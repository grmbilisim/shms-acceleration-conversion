
from fpdf import FPDF, HTMLMixin
import pandas as pd
import json 

class HTML2PDF(FPDF, HTMLMixin):
    def header(self):
        # Logo
        self.image(name='/home/grm/acc-data-conversion/shms-acceleration-conversion/convert_acc/resources/placeholder.png', x=10, y=6, w=45)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(120)
        # Title
        #self.cell(30, 10, 'Title', 1, 0, 'C')

        self.cell(w=60, h=10, txt='SUMMARY REPORT', border=0, ln=1, align='R')


        self.set_font('Arial', '', 7)
        self.set_text_color(r=128, g=128, b=128)
        self.cell(w=180, h=0, txt='assessment of impact due to building motion', border=0, ln=0, align='R')
        self.ln(1)
        self.line(5,25,205,25)
        self.ln(2)

        self.set_text_color(0,0,0)
        self.set_font('Arial', 'B', 8)
        self.cell(w=10, h=10, txt='EVENT:', border=0, ln=0, align='L')
        
        self.cell(w=1)
        self.set_text_color(0,0,255)
        self.cell(w=25, h=10, txt='TIMESTAMP PLACEHOLDER', border=0, ln=0, align='L')

        self.cell(w=100)
        self.set_text_color(0,0,0)
        self.cell(w=5, h=10, txt='SITE:', border=0, ln=0, align='L')
        self.cell(w=1)
        self.set_text_color(0,0,255)
        self.cell(w=37, h=10, txt='some building, Istanbul', border=0, ln=0, align='R')

        # Line break
        self.ln(2)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def html2pdf():

    df = pd.DataFrame({'a': [1,2,3], 'b':[4,5,6]})
    columns = df.columns.values
    html = '<table align="center" width="100%" border="0"><thead><tr bgcolor = "#E3E3E3">'
    for column in columns:
        html += '<th width="15%" height="35">{0}</th>'.format(column)
    html += '</thead><tbody>'

    s = df.to_json()
    jsonDict = json.loads(s)

    for x in range(0, len(jsonDict[columns[0]])):
        if x % 2 == 0:
            bgcolor = "#FFFFFF"
        else:
            bgcolor = "#F0F0F0"
        row = '<tr bgcolor ={0}>'.format(bgcolor)

        for c in columns:
            row += '<td align="center" height="35">{0}</td>'.format(jsonDict[c][str(x)])

        row += '</tr>'

        html += row

    html += '</tbody></table>'

    pdf = HTML2PDF()
    pdf.add_page()

    pdf.write_html(html)
    pdf.output('fpdf_example.pdf')


if __name__ == '__main__':
    html2pdf()