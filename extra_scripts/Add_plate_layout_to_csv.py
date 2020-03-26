
import openpyxl
import os
import shutil
import numpy as np

peptide_layout = np.array([['No Pep', 'CCPent', 'QLKEIA', 'CCHex2-I24K', 'CCHept-I24N', 'CCHept-L28W'],
                           ['GRP22', 'CCHex', 'NLKEIA', 'CCHept-L28K', 'CCHept-I24T', 'CCHept-L7Y'],
                           ['GRP35', 'CCHex2', 'CCHept-I17G-L21G', 'CCHept-I17H', 'CCHept-I24H', 'CCHept-L28Y'],
                           ['GRP46', 'CCHept', 'CCHept-I17A-L21A', 'CCHept-I17K-L24E', 'CCHept-L21S-I24S', 'CCHept-I24Y'],
                           ['GRP51', 'CCHept-I24D', 'CCHept-L14A', 'CCHept-L7K', 'CCHept-L21N-I24N', 'CCHept-L21S-I24Y'],
                           ['GRP52', 'CCHept-I24E', 'CCHept-L21A', 'CCHept-L21K-I24E', 'CCHept-L21T-I24T', 'CCHept-L21Y-I24S'],
                           ['GRP63', 'CCHept-I24K', 'CCHept-L21K', 'CCHept-L21E-I24K', 'CCHept-L21H-I24H', 'CCHept-I17T'],
                           ['GRP80', 'CCHept-I17K', 'CCPent-I24K', 'CCHept-I24S', 'CCHept-L7W', 'CCHept-I24P']])

input_dir = input('Specify input directory:\n')
new_dir = '{}/Reformatted_csvs/'.format(input_dir)
if os.path.isdir(new_dir):
    shutil.rmtree(new_dir)
os.mkdir(new_dir)

for csv in os.listdir(input_dir):
    if csv.endswith('.xlsx') and not 'plate_layout' in csv.lower():
        analyte = csv.split('_')[1]

        plate_layout = np.array([['', '1:6', '7:12', '13:18', '19:24'],
                                 ['A:H', analyte, 'Blank', analyte, 'Blank'],
                                 ['I:P', 'Blank', analyte, 'Blank', analyte]])

        new_csv = '{}/{}'.format(new_dir, '_'.join(csv.split('_')[1:]))
        print('Saving {}'.format(new_csv))
        if os.path.isfile(new_csv):
            raise Exception('{} already exists'.format(new_csv))

        orig_sheet = openpyxl.load_workbook(filename='{}/{}'.format(input_dir, csv))['End point']

        workbook = openpyxl.Workbook()
        # Copies End point worksheet from original csv file
        worksheet_1 = workbook.active
        worksheet_1.title = 'End point'
        for r in range (1, orig_sheet.max_row+1):
            for c in range (1, orig_sheet.max_column+1):
                cell = orig_sheet.cell(row=r, column=c)
                worksheet_1.cell(row=r, column=c).value = cell.value
        #Writes Protocol Information worksheet
        worksheet_2 = workbook.create_sheet(title='Protocol Information')
        worksheet_2.cell(row=2, column=1).value = 'Plate layout'
        for index, val in np.ndenumerate(plate_layout):
            r = index[0]
            c = index[1]
            worksheet_2.cell(row=r+3, column=c+1).value = val
        worksheet_2.cell(row=7, column=1).value = 'Peptide layout'
        for index, val in np.ndenumerate(peptide_layout):
            r = index[0]
            c = index[1]
            worksheet_2.cell(row=r+8, column=c+1).value = val
        # Saves workbook
        workbook.save(filename=new_csv)
