"""
Functions for assembling a docstring from various sources.
"""

import re

def paragraphs(linelist):
    plist = []
    newlinelist = []
    for line in linelist:
        line = line.strip()
        if line:
            newlinelist.append(line)
        elif newlinelist:
            plist.append(newlinelist)
            newlinelist = []
    if newlinelist:
        plist.append(newlinelist)
    return plist

def fix_outputs_doc(lines):
    """
    Convert a list of lines from the Matlab OUTPUTS section
    into a list suitable for the numpydoc Returns section.
    """
    for i, line in enumerate(list(lines)):
        match = re.search(r'(.*)\[(.*)\]', line)
        if match is not None:
            units = match.group(2).strip()
            remainder = match.group(1).strip()
            lines[i] = remainder
        else:
            lines[i] = line.strip()

    outname, remainder = lines[0].split('=')
    outlines = ['%s : array-like, %s' % (outname.strip(), units)]
    # FIXME: we need to assemble the rest into a single line
    # and then break it into lines of appropriate length.
    for line in lines[1:]:
        if line:
            outlines.append('    ' + line)
    return outlines

def docstring_from_sections(sections):
    """
    sections is a dictionary containing numpydoc text sections,
    without their headers.  Everything above the Parameters is
    considered Head; it does not have to follow the standard of
    having a single line "short summary", etc.  Each section
    must be a list of lines without newlines, and with
    indentation only relative to the edge of the docstring.

    We start with only 3 sections: Head, Parameters, and Returns;
    others may be added as options later.

    """
    doclines = ['']
    doclines.extend(sections['Head'])
    doclines.append('')
    doclines.append('Parameters')
    doclines.append('----------')
    doclines.extend(sections['Parameters'])
    doclines.append('')
    doclines.append('Returns')
    doclines.append('-------')
    doclines.extend(sections['Returns'])

    for i, line in enumerate(list(doclines)):
        if line:
            doclines[i] = '    %s\n' % line
        else:
            doclines[i] = '\n'

    return ''.join(doclines)
