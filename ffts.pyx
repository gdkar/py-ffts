
from libc.stdio  cimport *
from libc.stdlib cimport *
from libc.math   cimport *
from libc.stdint cimport *
from libc.stddef cimport *

from cython  cimport view
from cpython cimport array as carray
from cpython cimport buffer as cbuffer

#define POSITIVE_SIGN 1
#define NEGATIVE_SIGN -1
cdef int POSITIVE_SIGN = -1
cdef int NEGATIVE_SIGN = -1
cdef extern from "ffts.h" nogil:
   cdef struct _ffts_plan_t:
      pass
   ctypedef _ffts_plan_t ffts_plan_t;

   cdef ffts_plan_t *ffts_init_1d(size_t N, int sign);
   cdef ffts_plan_t *ffts_init_2d(size_t N1, size_t N2, int sign);
   cdef ffts_plan_t *ffts_init_nd(int rank, size_t *Ns, int sign);

# For real transforms, sign == -1 implies a real-to-complex forwards tranform,
# and sign == 1 implies a complex-to-real backwards transform
# The output of a real-to-complex transform is N/2+1 complex numbers, where the
# redundant outputs have been omitted.
   cdef ffts_plan_t *ffts_init_1d_real(size_t N, int sign);
   cdef ffts_plan_t *ffts_init_2d_real(size_t N1, size_t N2, int sign);
   cdef ffts_plan_t *ffts_init_nd_real(int rank, size_t *Ns, int sign);

   cdef void ffts_execute(ffts_plan_t * , const void *_in, void *_out);
   cdef void ffts_free(ffts_plan_t *);


cdef class Plan( object ):
    cdef ffts_plan_t *plan;
    cdef readonly carray.array  nd
    cdef readonly carray.array  in_size
    cdef readonly carray.array  out_size
    cdef readonly int  rank
    cdef          int  sign
    cdef readonly int  inverse 
    def __cinit__ ( self, N, inverse = False):
        if not hasattr(N,'__len__'):
            N=[N]
        self.rank = len(N)
        self.nd       = carray.array('L',N)
        self.in_size  = carray.array('L',N)
        self.out_size = carray.array('L',N)
        self.inverse = inverse;
        cdef size_t *nd_ptr
        for i in range(self.rank):
            self.in_size[i] = self.out_size[i] = 2*N[i]
            self.nd[i] = N[i]
        if inverse:
            self.out_size,self.in_size=self.in_size,self.out_size 
        self.sign = 1 if self.inverse else -1;
        if self.rank == 1:
            self.plan  = ffts_init_1d ( self.nd[0], self.sign )
        elif self.rank == 2:
            self.plan = ffts_init_2d ( self.nd[0],self.nd[1],self.sign )
        else:
            nd_ptr = <size_t*>self.nd.data.as_voidptr
            self.plan = ffts_init_nd ( self.rank, nd_ptr, self.sign )
    def __dealloc__ ( self ):
        self.nd[:]      = None
        self.in_size[:] = None
        self.out_size[:]= None
        self.rank = 0
        ffts_free(self.plan);
        self.plan = NULL
    cdef void _execute ( self, void *_in, void *_out):
        ffts_execute(self.plan,_in,_out)
    def execute (self, object _in, object _out):
        cdef cbuffer.Py_buffer in_buf
        cdef cbuffer.Py_buffer out_buf
        if cbuffer.PyObject_GetBuffer(_in, &in_buf,cbuffer.PyBUF_WRITABLE|cbuffer.PyBUF_CONTIG) < 0:
            return
        if cbuffer.PyObject_GetBuffer(_out,&out_buf,cbuffer.PyBUF_WRITABLE|cbuffer.PyBUF_CONTIG) < 0:
            cbuffer.PyBuffer_Release(&in_buf)
            return
        self._execute(in_buf.buf,out_buf.buf)
        cbuffer.PyBuffer_Release(&in_buf)
        cbuffer.PyBuffer_Release(&out_buf)

cdef class PlanReal( Plan):
    def __cinit__ ( self, N, inverse = False):
        if not hasattr(N,'__len__'):
            N=[N]
        self.rank = len(N)
        self.nd       = carray.array('L',N)
        self.in_size  = carray.array('L',N)
        self.out_size = carray.array('L',N)
        self.inverse = inverse;
        cdef size_t *nd_ptr
        for i in range(self.rank):
            self.nd[i]       = N[i]
            self.in_size[i]  = N[i]
            self.out_size[i] = 2*(N[i]/2+1)
        if inverse:
            self.out_size,self.in_size=self.in_size,self.out_size 
        self.sign = 1 if self.inverse else -1;
        if self.rank == 1:
            self.plan  = ffts_init_1d_real ( self.nd[0], self.sign )
        elif self.rank == 2:
            self.plan = ffts_init_2d_real ( self.nd[0],self.nd[1],self.sign )
        else:
            nd_ptr = <size_t*>self.nd.data.as_voidptr
            self.plan = ffts_init_nd_real ( self.rank, nd_ptr, self.sign )
    def __dealloc__ ( self ):
        self.nd[:]      = None
        self.in_size[:] = None
        self.out_size[:]= None
        self.rank = 0
        ffts_free(self.plan);
        self.plan = NULL
    cdef void _execute ( self, void *_in, void *_out):
        ffts_execute(self.plan,_in,_out)
    def execute (self, object _in, object _out):
        cdef cbuffer.Py_buffer in_buf
        cdef cbuffer.Py_buffer out_buf
        if cbuffer.PyObject_GetBuffer(_in, &in_buf,cbuffer.PyBUF_WRITABLE|cbuffer.PyBUF_CONTIG) < 0:
            return
        if cbuffer.PyObject_GetBuffer(_out,&out_buf,cbuffer.PyBUF_WRITABLE|cbuffer.PyBUF_CONTIG) < 0:
            cbuffer.PyBuffer_Release(&in_buf)
            return
        self._execute(in_buf.buf,out_buf.buf)
        cbuffer.PyBuffer_Release(&in_buf)
        cbuffer.PyBuffer_Release(&out_buf)
