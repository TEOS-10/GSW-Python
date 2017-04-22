"""
Functions for assembling a docstring from various sources.

As a temporary measure, we are getting the numpydoc 'Returns'
section by manipulating the Matlab 'OUTPUTS' block, rather than
by using a dictionary of blocks as we are for 'Parameters'.
"""

import re

def paragraphs(linelist):
    """
    Break a list of lines at blank lines into a list of line-lists.
    """
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


def fix_one_output(lines):
    """
    Convert a list of lines for a single output variable from
    the Matlab OUTPUTS section into a list suitable for the
    numpydoc Returns section.
    """
    units = ''  # Probably a hack; look into a better workaround.

    lines_orig = lines.copy()
    lines = []

    for line in lines_orig:
        match = re.search(r'(.*)\[(.*)\]', line)
        if match is not None:
            units = match.group(2).strip()
            remainder = match.group(1).strip()
            if 'has units of' not in remainder:
                lines.append(remainder)
        else:
            line = line.strip()
            if 'has units of' in line or line.startswith('Note'):
                break
            lines.append(line)

    outname, remainder = lines[0].split('=')
    outlines = ['%s : array-like, %s' % (outname.strip(), units)]
    remainder = remainder.strip()
    if remainder:
        outlines.append('    ' + remainder)
    # FIXME: we need to assemble the rest into a single line
    # and then break it into lines of appropriate length.
    for line in lines[1:]:
        if line:
            outlines.append('    ' + line)
    return outlines

def fix_outputs_doc(lines):
    """
    Convert a list of lines from the Matlab OUTPUTS section
    into a list suitable for the numpydoc Returns section,
    handling multiple output variables if present.

    Exception: we don't support the 'in_ocean' return, so
    it is filtered out here.
    """
    pat = r'^\s*(\w+)\s+='
    istarts = []
    for i, line in enumerate(lines):
        m = re.match(pat, line)
        if m is not None and m.groups()[0] == 'in_ocean':
            lines = lines[:i]
            break
        if m is not None:
            istarts.append(i)
    iends = istarts[1:] + [len(lines)]
    outlines = []
    for i0, i1 in zip(istarts, iends):
        outlines.extend(fix_one_output(lines[i0:i1]))
    # Add a blank line if needed, as in the case where we toss the
    # in_ocean chunk.  (Maybe we are losing this blank line somewhere else...)
    if outlines[-1]:
        outlines.append('')
    return outlines



def docstring_from_sections(sections):
    """
    sections is a dictionary containing numpydoc text sections,
    without their headers.  Everything above the Parameters is
    considered Head; it does not have to follow the standard of
    having a single line "short summary", etc.  Each section
    must be a list of lines without newlines, and with
    indentation only relative to the edge of the docstring.

    Only the Head is required.

    """
    doclines = ['']
    doclines.extend(sections['Head'])
    for name in ['Parameters',
                 'Returns',
                 'Other Parameters',
                 'Raises',
                 'See Also',
                 'Notes',
                 'References',
                 'Examples',
                 ]:
        if name in sections:
            doclines.extend(['',
                             name,
                             '-' * len(name),])
            doclines.extend(sections[name])

    for i, line in enumerate(list(doclines)):
        if line:
            doclines[i] = '    %s\n' % line.rstrip()
        else:
            doclines[i] = '\n'

    return ''.join(doclines)
