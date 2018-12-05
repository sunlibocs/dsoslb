/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

 
#include "util/NumType.h"

namespace dso
{
struct RawResidualJacobian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// ================== new structure: save independently =============.
	//对应r21，1x8，这里的r21是对于一个点，八个 pattern residual 组成的向量。
	VecNRf resF;

	// the two rows of d[x,y]/d[xi].
	//x21位姿的雅克比
	Vec6f Jpdxi[2];			// 2x6

	// the two rows of d[x,y]/d[C].
	//x2 相机内参的雅克比，C指相机内参[fx,fy,cx,cy]T
	VecCf Jpdc[2];			// 2x4

	// the two rows of d[x,y]/d[idepth].
	//host逆深度的雅克比
	Vec2f Jpdd;				// 2x1

	// the two columns of d[r]/d[x,y].
	//target 帧上(x2)的影像梯度相关
	VecNRf JIdx[2];			// 9x2   这里雅克布应该是注释有问题, 应该是8 ×2， 8代表8个pattern

	// = the two columns of d[r] / d[ab]

	//x21 光度雅克比
	VecNRf JabF[2];			// 9x2 同理8*2


	// = JIdx^T * JIdx (inner product). Only as a shorthand.
	//像素雅克比T× 像素雅克比 
	Mat22f JIdx2;				// 2x2
	// = Jab^T * JIdx (inner product). Only as a shorthand.
	//光度雅克比T × 像素雅克比
	Mat22f JabJIdx;			// 2x2
	// = Jab^T * Jab (inner product). Only as a shorthand.
	//光度雅克比T × 光度雅克比
	Mat22f Jab2;			// 2x2
	// https://www.cnblogs.com/JingeTU/p/8395046.html 中有详细的说明

};
}

