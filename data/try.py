# # user zhengkelong
class T():
    '''
   this is used to calcu
   '''
    a = 3

    def __init__(self):
        '''
        :param self:
        :return:
        '''
        # this

    def __str__(self):
        return '__str__'

    def __repr__(self):
        return '__repr__'

    def __get__(self, key, value):
        print(f'in:__get__')
        return None

    def __set__(self, key, value):
        print(f'in:__set__')

    def __getattr__(self, item):
        print(f'in:__getattr__')
        return None

    #
    def __setattr__(self, key, value):
        print(dir(super(T, self)))
        super(T, self).__setattr__(key, value)
        print(f'in:__setattr__')


a = T()
print(str(a))
print(repr(a))
print(a)
a.b = 3
c = a.b
print(dir(a), a.__dict__)
print(c)
print(f'this is {a.__class__.__mro__[1].__dict__}')


class B():
    a = T()
    print('i')

    def __init__(self):
        print('f')
        self.d = T()
        print('g')


b = B()
b.a
#
#
# class Philosopher:
#     def __init_subclass__(cls, /, default_name, **kwargs):
#         super().__init_subclass__(**kwargs)
#         cls.default_name = default_name
#
# b=Philosopher('3')
#


# defining a SuperClass
import itertools


#
# class SuperClass:
#
#     # defining __init_subclass__ method
#     def __init_subclass__(cls, **kwargs):
#         cls.default_name = "Inherited Class"
#         print('hia')
#
#
# # defining a SubClass
# class SubClass(SuperClass):
#     # an attribute of SubClass
#     print('ha')
#
#     def __init__(self):
#         self.default_name = "SubClass"
#         print(self.default_name)
#
#
# subclass = SubClass()
# print(subclass.default_name)
#
#
# class Test(object):
#
#     def __init_subclass__(cls, **kwargs):
#         print("__init_subclass__", cls, kwargs)
#
#
# class A(Test, name="张三", age=16):
#     pass
#
#
# from typing import Optional
#
#
# def main(a: Optional[str]):
#     print(a)
#
#
# class T:
#     def __init__(*args):
#         print(len(args))
#         self,a,c=args
#         print(type(args))
#         self.__name__='aja'
#     def q(self):
#         self.b=6
#         print(self.b)
# import operator
#
# a=[1]
# def m():
#     pass
# print(m.__name__)
# c=T(m,3)
#
# print(m.__name__)
# print(c.__name__)
# c.q()
# print(c.b)
#
#
#
#
#
# import matplotlib
#
# # a=(i for i in range(3))
# # print(a[0])
#
#
# class T:
#     def __int__(self):
#         pass
#
#     @staticmethod
#     def stat(*args):
#         print(f'the length of the args:{len(args)}')
#     @classmethod
#     def clas_method(*args): #需要一个cls
#         print(f'the length of the args:{len(args)}')
#
#
# T.stat()
# T.clas_method()
#
# # help(itertools.count.__new__)
#
#
#
# import docopt
#
#
# import logging
#
# logging.basicConfig(level=logging.INFO)
#
# class LoggedAccess:
#
#     def __set_name__(self, owner, name):
#         self.public_name = name
#         self.private_name = '_' + name
#
#     def __get__(self, obj, objtype=None):
#         value = getattr(obj, self.private_name)
#         logging.info('Accessing %r giving %r', self.public_name, value)
#         return value
#
#     def __set__(self, obj, value):
#         logging.info('Updating %r to %r', self.public_name, value)
#         setattr(obj, self.private_name, value)
#
# class Person:
#
#     name = LoggedAccess()                # First descriptor instance
#     age = LoggedAccess()                 # Second descriptor instance
#
#     def __init__(self, name, age):
#         self.name = name                 # Calls the first descriptor
#         self.age = age                   # Calls the second descriptor
#
#     def birthday(self):
#
#         self.age += 1
#
#
#
# a=Person(3,4)


class R:
    def __init__(self):
        self.b = 3
        print(f'this is the R:{type(self)}')


class B(R):
    def __init__(self):
        super(B, self).__init__()
        self.b = 4

    def q(self):
        pass


b = B()
# super(B,b).__setattr__('a',8)
# print(b.a)
print(b.__dict__)
print(super(B, b).__dict__)
print(b.__class__.__mro__[1])

print(dir(R))
print(dir(B))

print(type(R.__dict__['__dict__']))
print(B.__dict__)

print(b.__class__.__mro__[1] is R)
print(R.__dict__)


class F:
    def __init__(self):
        pass

    def __set__(self, key, value):
        print('this is __set__()')

    def __get__(self, key, value):
        print('this is __get__')


class E:
    def __init__(self):
        pass


print(E.__dict__)
print(dir(E))


class Q:
    a = 5
    b = 6
    c = F()

    def __init__(self):
        super(Q, self).__setattr__('a', 3)

    # def  __setattr__(self, key, value):
    #     print('__setattr__')


b = Q()
b.a = 9  # __setattr__
Q.a = 7
b.b = 9  # __setattr__,__dict__
b.c = 7  # __setattr__,__dict__
print(b.a)
Q.c = 4
print(b.__dict__)
print(Q.__dict__)
print(Q.a)
print(b.b)
# print(dir(Q))
print(b.c)  # 调用__get__


#
# print(dir(b))
#
# print(Q.__dict__)
# print(b.__dict__)
# c=Q()
#
# print(c.__dict__)


class A:
    a = 3

    def __init__(self):
        pass

print(A.__dict__)

class B(A):
    b = 4

    def __init__(self):
        super(B, self).__init__()
        pass
print(B.__dict__)
a=B()
print(A.a)
print(a.a)
A.a=8
print(a.a)


print(dir(B))

print(id(a.a))
a.a=7

print(dir(a))