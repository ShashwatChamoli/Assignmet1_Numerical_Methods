// -*- C++ -*-
//===--------------------------- iostream ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_IOSTREAM
#define _LIBCPP_IOSTREAM

/*
    iostream synopsis

#include <ios>
#include <istream>
#include <ostream>
#include <streambuf>

namespace std {

extern istream cin;
extern ostream cout;
extern ostream cerr;
extern ostream clog;
extern wistream wcin;
extern wostream wcout;
extern wostream wcerr;
extern wostream wclog;

}  // std

*/

#include <__config>
#include <ios>
#include <istream>
#include <ostream>
#include <streambuf>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

extern _LIBCPP_FUNC_VIS istream cin;
extern _LIBCPP_FUNC_VIS ostream cout;
extern _LIBCPP_FUNC_VIS ostream cerr;
extern _LIBCPP_FUNC_VIS ostream clog;

#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
extern _LIBCPP_FUNC_VIS wistream wcin;
extern _LIBCPP_FUNC_VIS wostream wcout;
extern _LIBCPP_FUNC_VIS wostream wcerr;
extern _LIBCPP_FUNC_VIS wostream wclog;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_IOSTREAM