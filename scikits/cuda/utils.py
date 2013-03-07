#!/usr/bin/env python

"""
Utility functions.
"""

import re
import subprocess

try:
    import elftools
except ImportError:
    import re

    def get_soname(filename):
        """
        Retrieve SONAME of shared library.

        Parameters
        ----------
        filename : str
            Full path to shared library.

        Returns
        -------
        soname : str
            SONAME of shared library.

        Notes
        -----
        This function uses the `objdump` system command.
        
        """
        
        try:
            p = subprocess.Popen(['objdump', '-p', filename],
                                 stdout=subprocess.PIPE)
            out = p.communicate()[0]
        except:
            raise RuntimeError('error executing objdump')
        else:
            result = re.search('^\s+SONAME\s+(.+)$',out,re.MULTILINE)
            if result:
                return result.group(1)
            else:

                # No SONAME found:
                return ''

else:
    import ctypes
    import elftools.elf.elffile as elffile
    import elftools.construct.macros as macros
    import elftools.elf.structs as structs

    def get_soname(filename):
        """
        Retrieve SONAME of shared library.

        Parameters
        ----------
        filename : str
            Full path to shared library.

        Returns
        -------
        soname : str
            SONAME of shared library.

        Notes
        -----
        This function uses the pyelftools [ELF] package.

        References
        ----------
        .. [ELF] http://pypi.python.org/pypi/pyelftools
        
        """

        stream = open(filename, 'rb')
        f = elffile.ELFFile(stream)
        dynamic = f.get_section_by_name('.dynamic')
        dynstr = f.get_section_by_name('.dynstr')

        # Handle libraries built for different machine architectures:         
        if f.header['e_machine'] == 'EM_X86_64':
            st = structs.Struct('Elf64_Dyn',
                                macros.ULInt64('d_tag'),
                                macros.ULInt64('d_val'))
        elif f.header['e_machine'] == 'EM_386':
            st = structs.Struct('Elf32_Dyn',
                                macros.ULInt32('d_tag'),
                                macros.ULInt32('d_val'))
        else:
            raise RuntimeError('unsupported machine architecture')

        entsize = dynamic['sh_entsize']
        for k in xrange(dynamic['sh_size']/entsize):
            result = st.parse(dynamic.data()[k*entsize:(k+1)*entsize])

            # The following value for the SONAME tag is specified in elf.h:  
            if result.d_tag == 14:
                return dynstr.get_string(result.d_val)

        # No SONAME found:
        return ''

def find_lib_path(filename):
    """
    Find full path of a shared library.

    Parameter
    ---------
    filename : str
        Basename of library to search for.

    Returns
    -------
    path : str
        Full path to library.

    """

    p = subprocess.Popen(['/sbin/ldconfig', '-p'], stdout=subprocess.PIPE)
    output = p.communicate()[0]
    result = re.search('^\s*%s\s.*\=\>\s(.+)$' % filename, output, re.MULTILINE)
    if result:
        return result.group(1)
    else:
        raise RuntimeError('library not found')

