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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0,0), thisToNext(SE3())
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		points[lvl] = 0;
		numPoints[lvl] = 0;
	}

	JbBuffer = new Vec10f[ww*hh];
	JbBuffer_new = new Vec10f[ww*hh];


	frameID=-1;
	fixAffine=true;
	printDebug=false;

	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer()
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		if(points[lvl] != 0) delete[] points[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
}


//这个只有执行初始化的时候才会执行这个track，优化两针之间的位姿
//初始化成功后执行的是另一个

//第二帧来了以后进行初始化的track
//使用直接图像对齐，计算出两帧之间6个自由度的se3 转换，
//更新点的点的深度以及光度参数a, b
bool CoarseInitializer::trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
	newFrame = newFrameHessian;
	//输出frame
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushLiveFrame(newFrameHessian);

	int maxIterations[] = {5,5,10,30,50};


	alphaK = 2.5*2.5;//*freeDebugParam1*freeDebugParam1;
	alphaW = 150*150;//*freeDebugParam2*freeDebugParam2;
	regWeight = 0.8;//*freeDebugParam4;
	couplingWeight = 1;//*freeDebugParam5;
	//将所有点的信息-深度，hessian初始化
	if(!snapped)
	{
		thisToNext.translation().setZero();
		for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
		{
			int npts = numPoints[lvl];
			Pnt* ptsl = points[lvl];
			for(int i=0;i<npts;i++)
			{
				ptsl[i].iR = 1;
				ptsl[i].idepth_new = 1;
				ptsl[i].lastHessian = 0;
			}
		}
	}

	//ref到current　的se3转换
	SE3 refToNew_current = thisToNext;
	//仿射变换
	AffLight refToNew_aff_current = thisToNext_aff;

	//仿射变换
	//参数为当前的曝光参数/第一帧的曝光参数
	if(firstFrame->ab_exposure>0 && newFrame->ab_exposure>0)
		refToNew_aff_current = AffLight(logf(newFrame->ab_exposure /  firstFrame->ab_exposure),0); // coarse approximation.


	Vec3f latestRes = Vec3f::Zero();
	//从金字塔的最顶端开始循环
	//分辨率最小的那层开始做
	for(int lvl=pyrLevelsUsed-1; lvl>=0; lvl--)
	{
		//****srcLvl越大，图像的分辨率越小
		//使用当前层的点的深度信息去更新srcLvl－１层点的信息
		//之后再使用邻居点的深度更新当前srcLvl－１点的所有深度
		// 因为 lvl+1 其实信息是已经算出来了的
		if(lvl<pyrLevelsUsed-1)
			propagateDown(lvl+1);

		Mat88f H,Hsc; Vec8f b,bsc;
		//设置点深度，为临近10个点深度和的均值
		resetPoints(lvl);
		//计算残差和？H以及b
		Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
		//执行step，更新point的depth,energy等信息
		applyStep(lvl);

		float lambda = 0.1;
		float eps = 1e-4;
		int fails=0;
		//输出调试信息
		if(printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
					lvl, 0, lambda,
					"INITIA",
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					(resOld[0]+resOld[1]) / resOld[2],
					(resOld[0]+resOld[1]) / resOld[2],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() <<"\n";
		}

		int iteration=0;
		while(true)
		{
			Mat88f Hl = H;
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda);

			//两式相减（H - Hsc, b - bsc）求得 最后的H和 b
			//keynote相见 https://www.cnblogs.com/JingeTU/p/8297076.html
			Hl -= Hsc*(1/(1+lambda));
			Vec8f bl = b - bsc*(1/(1+lambda));

			//得到加权后的H1矩阵
			Hl = wM * Hl * wM * (0.01f/(w[lvl]*h[lvl]));
			//得到加权后的b1矩阵
			bl = wM * bl * (0.01f/(w[lvl]*h[lvl]));


			//论文中新的delta,用于更新迭代增量
			Vec8f inc;
			//如果修复仿射
			if(fixAffine)
			{
				// ldlt + solve 是使用奇异值分解求解方程 Hx = b
				inc.head<6>() = - (wM.toDenseMatrix().topLeftCorner<6,6>() * (Hl.topLeftCorner<6,6>().ldlt().solve(bl.head<6>())));
				inc.tail<2>().setZero();
			}
			//得到的delta增量
			else
				inc = - (wM * (Hl.ldlt().solve(bl)));	//=-H^-1 * b.

			//通过迭代增量更新当前的位姿信息
			//（但是呢，只有这一步被接收了才会用这个新的去更新refToNew_current）
			SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
			AffLight refToNew_aff_new = refToNew_aff_current;
			//更新这个仿射变换的参数
			//证明了inc 是六个自由度的se + 两个自由度的光度参数a 和 b
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			//利用迭代的增量和雅克比更新points中所有点的深度信息
			doStep(lvl, lambda, inc);


			Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
			//更新位姿和仿射变换参数后，再次计算残差，Ｈ,b
			//因为要判断这次迭代是否有效，所以要试一下这次迭代后的能量是否小于之前的能量
			//同时计算出Hnew bnew 用于下次迭代用（被accpet）
			Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
			Vec3f regEnergy = calcEC(lvl);

			//新的总能量
			float eTotalNew = (resNew[0]+resNew[1]+regEnergy[1]);
			//旧的总能量
			float eTotalOld = (resOld[0]+resOld[1]+regEnergy[0]);

			//新的比旧的小才会接收
			bool accept = eTotalOld > eTotalNew;

			if(printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						(accept ? "ACCEPT" : "REJECT"),
						sqrtf((float)(resOld[0] / resOld[2])),
						sqrtf((float)(regEnergy[0] / regEnergy[2])),
						sqrtf((float)(resOld[1] / resOld[2])),
						sqrtf((float)(resNew[0] / resNew[2])),
						sqrtf((float)(regEnergy[1] / regEnergy[2])),
						sqrtf((float)(resNew[1] / resNew[2])),
						eTotalOld / resNew[2],
						eTotalNew / resNew[2],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() <<"\n";
			}
			//如果接收迭代
			if(accept)
			{

				if(resNew[1] == alphaK*numPoints[lvl])
					snapped = true;
				//赋值为新的H b 位姿矩阵，仿射矩阵，残差，用于下次迭代用
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				//更新点的深度，能量，Hessian等信息
				applyStep(lvl);
				//使用邻居点的深度更新当前lvl点的所有深度
				optReg(lvl);
				lambda *= 0.5;
				fails=0;
				if(lambda < 0.0001) lambda = 0.0001;
			}
			//如果不接受则更新lambda
			else
			{
				fails++;
				lambda *= 4;
				if(lambda > 10000) lambda = 10000;
			}

			bool quitOpt = false;
			//是否结束迭代
			//条件为增量小于eps，迭代次数大于ｍａｘ，失败次数太多
			if(!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
			{
				Mat88f H,Hsc; Vec8f b,bsc;
				quitOpt = true;
			}

			//如果结束迭代
			if(quitOpt) break;
			iteration++;
		}//迭代结束，否则继续迭代

		//最近的残差，通过和最新的比较可以知道是否迭代方向为下降的
		latestRes = resOld;

	}//当前金字塔层的循环结束，进入下一层进行循环


	//使用运动模型？
	thisToNext = refToNew_current;
	thisToNext_aff = refToNew_aff_current;

	for(int i=0;i<pyrLevelsUsed-1;i++)
	//利用低一层（分辨率更大）点的深度更新高层点的深度
		propagateUp(i);




	frameID++;
	if(!snapped) snappedAt=0;

	if(snapped && snappedAt==0)
		snappedAt = frameID;



    debugPlot(0,wraps);



	return snapped && frameID > snappedAt+5;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    bool needCall = false;
    for(IOWrap::Output3DWrapper* ow : wraps)
        needCall = needCall || ow->needPushDepthImage();
    if(!needCall) return;


	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

	MinimalImageB3 iRImg(wl,hl);

	for(int i=0;i<wl*hl;i++)
		iRImg.at(i) = Vec3b(colorRef[i][0],colorRef[i][0],colorRef[i][0]);


	int npts = numPoints[lvl];

	float nid = 0, sid=0;
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(point->isGood)
		{
			nid++;
			sid += point->iR;
		}
	}
	float fac = nid / sid;



	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;

		if(!point->isGood)
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));

		else
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,makeRainbow3B(point->iR*fac));
	}


	//IOWrap::displayImage("idepth-R", &iRImg, false);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImage(&iRImg);
}

// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
Vec3f CoarseInitializer::calcResAndGS(
		int lvl, Mat88f &H_out, Vec8f &b_out,
		Mat88f &H_out_sc, Vec8f &b_out_sc,
		const SE3 &refToNew, AffLight refToNew_aff,
		bool plot)
{
	int wl = w[lvl], hl = h[lvl];
	//得到当前level的像素梯度
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
	Eigen::Vector3f* colorNew = newFrame->dIp[lvl];
	//将点反投影到空间中，之后在旋转到当前坐标系下
	Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
	Vec3f t = refToNew.translation().cast<float>();
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

	//得到焦距，主点位置等参数
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];


	Accumulator11 E;
	acc9.initialize();
	E.initialize();

	//得到当前层的点的数目，以及所有点
	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	for(int i=0;i<npts;i++)
	{

		Pnt* point = ptsl+i;

		point->maxstep = 1e10;
		//如果为坏点
		if(!point->isGood)
		{
			E.updateSingle((float)(point->energy[0]));
			point->energy_new = point->energy;
			point->isGood_new = false;
			continue;
		}

        VecNRf dp0;
        VecNRf dp1;
        VecNRf dp2;
        VecNRf dp3;
        VecNRf dp4;
        VecNRf dp5;
        VecNRf dp6;
        VecNRf dp7;
        VecNRf dd;
        VecNRf r;
		//每次开始前，将上一次的置为0
		JbBuffer_new[i].setZero();

		// sum over all residuals.
		bool isGood = true;
		float energy=0;
		//patternNum = 8
		//对pattern中的每个点计算雅克比，并且加到对应的位置

		//acc9与公式中的Hx21x21,JTx21r21相关，

		//acc9SC与公式中的HTρx21H−1ρρHρx21,HTρx21H−1ρρJTρr21相关。

		//JbBuffer_new在 idx pattern 循环内，
		//分别对每点的8个 pattern 的JTx21Jρ,JTρr21,JρJTρ进行累加
		for(int idx=0;idx<patternNum;idx++)
		{
			//得到pattern中的第一个pixel的坐标
			int dx = patternP[idx][0];
			int dy = patternP[idx][1];

			//得到点在当前level，中的3D坐标
			//这里 Ki是反投影并且在当前坐标系下的3D位置，平移是一样的
			Vec3f pt = RKi * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;
			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			//得到在图像中的成像位置
			float Ku = fxl * u + cxl;
			float Kv = fyl * v + cyl;
			//得到深度
			float new_idepth = point->idepth_new/pt[2];

			if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0))
			{
				isGood = false;
				break;
			}
			//colorNew（dip） 为当前层的像素灰度和梯度，得到（Ku, Kv）这个点的梯度（用到了周围的插值）
			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
			//Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

			//float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
			//得到在 colorRef 中的像素灰度和梯度，上面那个hitColor是当前的image的
			float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl);

			if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
			{
				isGood = false;
				break;
			}

			//得到残差，当前的灰度 - 经过ref frame中像素经过仿射变换后的灰度
			float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
			//使用H胡贝尔函数
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			//得到胡贝尔处理后的能量残差
			energy += hw *residual*residual*(2-hw);

			//得到平移长度在图像中的投影长度？
			float dxdd = (t[0]-t[2]*u)/pt[2];
			float dydd = (t[1]-t[2]*v)/pt[2];

			if(hw < 1) hw = sqrtf(hw);
			//插值？
			float dxInterp = hw*hitColor[1]*fxl;
			float dyInterp = hw*hitColor[2]*fyl;
			//parten 内每个点都要计算
			dp0[idx] = new_idepth*dxInterp;
			dp1[idx] = new_idepth*dyInterp;
			dp2[idx] = -new_idepth*(u*dxInterp + v*dyInterp);
			dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp;
			dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp;
			dp5[idx] = -v*dxInterp + u*dyInterp;
			dp6[idx] = - hw*r2new_aff[0] * rlR;
			dp7[idx] = - hw*1;
			//dd[idx]为一个浮点数，dd存储了很多歌浮点数
			dd[idx] = dxInterp * dxdd  + dyInterp * dydd;
			r[idx] = hw*residual;

			float maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm();
			if(maxstep < point->maxstep) point->maxstep = maxstep;

			// immediately compute dp*dd' and dd*dd' in JbBuffer1.

			//JbBuffer_new 中要累加三个雅克比，依次为
			//帧的雅克比*点的雅克比
			//点的雅克比×r21
			//点的雅克比×点的雅克比

			
			//(帧之间的雅克比,光度雅克比) × 点的雅克比，因为真的雅克比为1*6维，点的雅克比为1维
			JbBuffer_new[i][0] += dp0[idx]*dd[idx];
			JbBuffer_new[i][1] += dp1[idx]*dd[idx];
			JbBuffer_new[i][2] += dp2[idx]*dd[idx];
			JbBuffer_new[i][3] += dp3[idx]*dd[idx];
			JbBuffer_new[i][4] += dp4[idx]*dd[idx];
			JbBuffer_new[i][5] += dp5[idx]*dd[idx];
			JbBuffer_new[i][6] += dp6[idx]*dd[idx];
			JbBuffer_new[i][7] += dp7[idx]*dd[idx];
			//点的雅克比 × r21
			JbBuffer_new[i][8] += r[idx]*dd[idx];
			//点的雅克比和点的雅克比的Jp * Jp ^ T
			JbBuffer_new[i][9] += dd[idx]*dd[idx];
		}//每个partten的循环执行完毕

		if(!isGood || energy > point->outlierTH*20)
		{
			E.updateSingle((float)(point->energy[0]));
			point->isGood_new = false;
			point->energy_new = point->energy;
			continue;
		}


		// add into energy.
		//把一个能量值添加进来，SSEData为所有能量值的求和
		E.updateSingle(energy);
		point->isGood_new = true;
		point->energy_new[0] = energy;

		// update Hessian matrix.
		//对每一个pattern的八个点的dp值进行累加
		//_mm_load_ps一次取四个，所以i+=4

		//这里acc为1×8矩阵 [J21,r] 
		//accT * acc = acc.H
		for(int i=0;i+3<patternNum;i+=4)
			acc9.updateSSE(
					_mm_load_ps(((float*)(&dp0))+i),
					_mm_load_ps(((float*)(&dp1))+i),
					_mm_load_ps(((float*)(&dp2))+i),
					_mm_load_ps(((float*)(&dp3))+i),
					_mm_load_ps(((float*)(&dp4))+i),
					_mm_load_ps(((float*)(&dp5))+i),
					_mm_load_ps(((float*)(&dp6))+i),
					_mm_load_ps(((float*)(&dp7))+i),
					_mm_load_ps(((float*)(&r))+i));

		//实际上下面这个for循环都不会被执行　如果i = patternNum　＝　８
		for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			acc9.updateSingle(
					(float)dp0[i],(float)dp1[i],(float)dp2[i],(float)dp3[i],
					(float)dp4[i],(float)dp5[i],(float)dp6[i],(float)dp7[i],
					(float)r[i]);


	}//对外围所有点的循环完毕

	E.finish();
	acc9.finish();


	// calculate alpha energy, and decide if we cap it.
	Accumulator11 EAlpha;
	EAlpha.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		//如果不是新的好的点
		if(!point->isGood_new)
		{
			E.updateSingle((float)(point->energy[1]));
		}
		else
		{//
			point->energy_new[1] = (point->idepth_new-1)*(point->idepth_new-1);
			E.updateSingle((float)(point->energy_new[1]));
		}
	}
	EAlpha.finish();
	float alphaEnergy = alphaW*(EAlpha.A + refToNew.translation().squaredNorm() * npts);

	//printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);


	// compute alpha opt.
	float alphaOpt;
	if(alphaEnergy > alphaK*npts)
	{
		alphaOpt = 0;
		alphaEnergy = alphaK*npts;
	}
	else
	{
		alphaOpt = alphaW;
	}


	acc9SC.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
			continue;

		point->lastHessian_new = JbBuffer_new[i][9];

		JbBuffer_new[i][8] += alphaOpt*(point->idepth_new - 1);
		JbBuffer_new[i][9] += alphaOpt;

		if(alphaOpt==0)
		{
			JbBuffer_new[i][8] += couplingWeight*(point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		}

		JbBuffer_new[i][9] = 1/(1+JbBuffer_new[i][9]);
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();


	//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
	
	//这里.H是求J^T * J

	//只用左上角第一块，相邻的第二块
	H_out = acc9.H.topLeftCorner<8,8>();// / acc9.num;
	b_out = acc9.H.topRightCorner<8,1>();// / acc9.num;
	H_out_sc = acc9SC.H.topLeftCorner<8,8>();// / acc9.num;
	b_out_sc = acc9SC.H.topRightCorner<8,1>();// / acc9.num;



	H_out(0,0) += alphaOpt*npts;
	H_out(1,1) += alphaOpt*npts;
	H_out(2,2) += alphaOpt*npts;

	Vec3f tlog = refToNew.log().head<3>().cast<float>();
	b_out[0] += tlog[0]*alphaOpt*npts;
	b_out[1] += tlog[1]*alphaOpt*npts;
	b_out[2] += tlog[2]*alphaOpt*npts;





	return Vec3f(E.A, alphaEnergy ,E.num);
}

float CoarseInitializer::rescale()
{
	float factor = 20*thisToNext.translation().norm();
//	float factori = 1.0f/factor;
//	float factori2 = factori*factori;
//
//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
//	{
//		int npts = numPoints[lvl];
//		Pnt* ptsl = points[lvl];
//		for(int i=0;i<npts;i++)
//		{
//			ptsl[i].iR *= factor;
//			ptsl[i].idepth_new *= factor;
//			ptsl[i].lastHessian *= factori2;
//		}
//	}
//	thisToNext.translation() *= factori;

	return factor;
}


Vec3f CoarseInitializer::calcEC(int lvl)
{
	if(!snapped) return Vec3f(0,0,numPoints[lvl]);
	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(!point->isGood_new) continue;
		float rOld = (point->idepth-point->iR);
		float rNew = (point->idepth_new-point->iR);
		E.updateNoWeight(Vec2f(rOld*rOld,rNew*rNew));

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);
}

//使用邻居点的深度更新当前lvl点的所有深度
void CoarseInitializer::optReg(int lvl)
{
	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	if(!snapped)
	{
		for(int i=0;i<npts;i++)
			ptsl[i].iR = 1;
		return;
	}


	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood) continue;

		float idnn[10];
		int nnn=0;
		//得到相邻点的坐标，之后存储在idnn中
		for(int j=0;j<10;j++)
		{
			if(point->neighbours[j] == -1) continue;
			Pnt* other = ptsl+point->neighbours[j];
			if(!other->isGood) continue;
			idnn[nnn] = other->iR;
			nnn++;
		}

		if(nnn > 2)
		{
			//得到距离为中位数的那个点
			std::nth_element(idnn,idnn+nnn/2,idnn+nnn);
			//使用中位数点的深度更新当前点的深度
			point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2];
		}
	}

}


//利用低一层（分辨率更大）点的深度更新高层点的深度
void CoarseInitializer::propagateUp(int srcLvl)
{
	assert(srcLvl+1<pyrLevelsUsed);
	// set idepth of target

	int nptss= numPoints[srcLvl];
	int nptst= numPoints[srcLvl+1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl+1];

	// set to zero.
	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		parent->iR=0;
		parent->iRSumNum=0;
	}

	//使用下层点的深度更新上一层（父节点）的点的深度
	for(int i=0;i<nptss;i++)
	{
		Pnt* point = ptss+i;
		if(!point->isGood) continue;

		Pnt* parent = ptst + point->parent;
		parent->iR += point->iR * point->lastHessian;
		parent->iRSumNum += point->lastHessian;
	}

	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		if(parent->iRSumNum > 0)
		{
			parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
			parent->isGood = true;
		}
	}

	optReg(srcLvl+1);
}
//****srcLvl越大，图像的分辨率越小
//使用当前层的点的深度信息去更新srcLvl－１层点的信息
//之后再使用邻居点的深度更新当前srcLvl－１点的所有深度
void CoarseInitializer::propagateDown(int srcLvl)
{
	assert(srcLvl>0);
	// set idepth of target
	//
	int nptst= numPoints[srcLvl-1];
	//当前层的所有点
	Pnt* ptss = points[srcLvl];
	//低一层的所有的点，之后再对其求得parent就相当于得到在srcLvl的位置了
	//再用这个位置去更新
	Pnt* ptst = points[srcLvl-1];
	
	for(int i=0;i<nptst;i++)
	{
		//低一层的节点作为子节点
		Pnt* point = ptst+i;
		//当前层的节点作为父点
		Pnt* parent = ptss+point->parent;

		if(!parent->isGood || parent->lastHessian < 0.1) continue;
		//如果点不好，使用父节点更新srcLvl这层的点
		if(!point->isGood)
		{
			point->iR = point->idepth = point->idepth_new = parent->iR;
			point->isGood=true;
			point->lastHessian=0;
		}
		//否则，使用父节点的深度更新当前
		else
		{
			float newiR = (point->iR*point->lastHessian*2 + parent->iR*parent->lastHessian) / (point->lastHessian*2+parent->lastHessian);
			point->iR = point->idepth = point->idepth_new = newiR;
		}
	}
	//使用邻居点的深度更新当前lvl点的所有深度
	optReg(srcLvl-1);
}


void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f* dINew_l = data[lvl];
		Eigen::Vector3f* dINew_lm = data[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
				dINew_l[x + y*wl][0] = 0.25f * (dINew_lm[2*x   + 2*y*wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1][0] +
													dINew_lm[2*x   + 2*y*wlm1+wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1+wlm1][0]);

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			dINew_l[idx][1] = 0.5f*(dINew_l[idx+1][0] - dINew_l[idx-1][0]);
			dINew_l[idx][2] = 0.5f*(dINew_l[idx+wl][0] - dINew_l[idx-wl][0]);
		}
	}
}

//首先得到满足梯度条件的用于生成map的pixel位置
//根据选出来的pixel生成对应的point,并且设置这些点的初始化参数
//对points中的每一个点，得到离它最近的10个点
//并且得到该point在下一层（分辨率更小）中的估计位置，并且得到离这个估计位置最近的点作为parent	
void CoarseInitializer::setFirst(CalibHessian* HCalib, FrameHessian* newFrameHessian)
{

	makeK(HCalib);
	firstFrame = newFrameHessian;

	//w[0] 应该是图像的宽度
	PixelSelector sel(w[0],h[0]);

	float* statusMap = new float[w[0]*h[0]] ;
	bool* statusMapB = new bool[w[0]*h[0]];

	float densities[] = {0.03,0.05,0.15,0.5,1};
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		sel.currentPotential = 3;
		int npts;
		if(lvl == 0)
			//得到用于生成map的，候选的pixel，结果存储在statusMap中,密度(个数为)densities[lvl]*w[0]*h[0]
			npts = sel.makeMaps(firstFrame, statusMap,densities[lvl]*w[0]*h[0],1,false,2);
		else
			//根据sparsityFactor，以及计算好的像素梯度grads，来选择地图点并且存储在map中
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);
		//Pnt* points[PYR_LEVELS]指针数组
		if(points[lvl] != 0) delete[] points[lvl];
		points[lvl] = new Pnt[npts];

		// set idepth map to initially 1 everywhere.
		int wl = w[lvl], hl = h[lvl];
		//points[lvl] 是当前level首地址的指针
		Pnt* pl = points[lvl];
		int nl = 0;
		//根据选出来的pixel生成对应的point,并且设置这些点的初始化参数
		for(int y=patternPadding+1;y<hl-patternPadding-2;y++)
		for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
		{
			//if(x==2) printf("y=%d!\n",y);
			//当前statusmap中提示像素满足条件
			if((lvl!=0 && statusMapB[x+y*wl]) || (lvl==0 && statusMap[x+y*wl] != 0))
			{
				//assert(patternNum==9);
				//设置当前pixel的状态
				pl[nl].u = x+0.1;
				pl[nl].v = y+0.1;
				pl[nl].idepth = 1;
				pl[nl].iR = 1;
				pl[nl].isGood=true;
				pl[nl].energy.setZero();
				pl[nl].lastHessian=0;
				pl[nl].lastHessian_new=0;
				pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];
				//得到存储梯度的指针
				Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
				float sumGrad2=0;
				//对pattern内的所有pixel的梯度求和
				for(int idx=0;idx<patternNum;idx++)
				{
					int dx = patternP[idx][0];
					int dy = patternP[idx][1];
					float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
					sumGrad2 += absgrad;
				}

//				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
//				pl[nl].outlierTH = patternNum*gth*gth;
				//设置outlier的阈值？
				pl[nl].outlierTH = patternNum*setting_outlierTH;
				//
				nl++;
				assert(nl <= npts);
			}
		}

		//点的个数
		numPoints[lvl]=nl;
	}
	delete[] statusMap;
	delete[] statusMapB;

	//对points中的每一个点，得到离它最近的10个点
	//并且得到该point在下一层（分辨率更小）中的估计位置，并且得到离这个估计位置最近的点作为parent
	makeNN();

	thisToNext=SE3();
	snapped = false;
	frameID = snappedAt = 0;

	for(int i=0;i<pyrLevelsUsed;i++)
		dGrads[i].setZero();

}

//设置点深度，为临近10个点深度和的均值
void CoarseInitializer::resetPoints(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		pts[i].energy.setZero();
		pts[i].idepth_new = pts[i].idepth;

		//lvl为最外层，或者点的状态为不好
		if(lvl==pyrLevelsUsed-1 && !pts[i].isGood)
		{
			float snd=0, sn=0;
			//得到最临近的10个点的深度和
			for(int n = 0;n<10;n++)
			{
				if(pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
				snd += pts[pts[i].neighbours[n]].iR;
				sn += 1;
			}
			//点深度，为临近10个点深度和的均值
			if(sn > 0)
			{
				pts[i].isGood=true;
				pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd/sn;
			}
		}
	}
}

//利用迭代的增量和雅克比更新points中所有点的深度信息
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
{

	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood) continue;


		//利用舒尔补计算出帧和帧之间的更新（方程的第二个项）以后
		//代入第一项，求得点的更新
		//公式为 δρ=−H−1ρρ(JTρr21+Hρx21δx21)

		//其中JbBuffer[i][9]为点的雅克比^T * 点的雅克比，即第一个 −H−1ρρ
		// b 为 JTρr21+Hρx21δx21

		//参考 https://www.cnblogs.com/JingeTU/p/8297076.html

		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		//得到点的深度的需要更新的值
		float step = - b * JbBuffer[i][9] / (1+lambda);


		float maxstep = maxPixelStep*pts[i].maxstep;
		if(maxstep > idMaxStep) maxstep=idMaxStep;

		if(step >  maxstep) step = maxstep;
		if(step < -maxstep) step = -maxstep;

		//梯度下降法后的新的深度
		float newIdepth = pts[i].idepth + step;
		if(newIdepth < 1e-3 ) newIdepth = 1e-3;
		if(newIdepth > 50) newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}

}
//更新点的深度，能量，Hessian等信息
void CoarseInitializer::applyStep(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood)
		{
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

//生成相机投影参数
void CoarseInitializer::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	//焦距和主点参数位置
	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();
	
	//生成对应层的焦距和主点参数
	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		//得到投影矩阵K
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		// Ki 为反投影矩阵
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}




//对points中的每一个点，得到离它最近的10个点
//并且得到该point在下一层（分辨率更小）中的估计位置，并且得到离这个估计位置最近的点作为parent	
void CoarseInitializer::makeNN()
{
	const float NNDistFactor=0.05;

	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
			FLANNPointcloud,2> KDTree;

	// build indices
	FLANNPointcloud pcs[PYR_LEVELS];
	KDTree* indexes[PYR_LEVELS];
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		indexes[i]->buildIndex();
	}

	const int nn=10;

	// find NN & parents
	//对每一层
	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	{
		//得到点
		Pnt* pts = points[lvl];
		//当前层的点的个数
		int npts = numPoints[lvl];
		//大小为10
		int ret_index[nn];
		float ret_dist[nn];
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for(int i=0;i<npts;i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			//坐标
			Vec2f pt = Vec2f(pts[i].u,pts[i].v);
			//得到10个距离最近的Neighbours
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			int myidx=0;
			float sumDF = 0;
			//生成和Nieghbours的距离
			//并且存储在pts的十个最临近的邻居中
			for(int k=0;k<nn;k++)
			{
				//
				pts[i].neighbours[myidx]=ret_index[k];
				float df = expf(-ret_dist[k]*NNDistFactor);
				sumDF += df;
				pts[i].neighboursDist[myidx]=df;
				assert(ret_index[k]>=0 && ret_index[k] < npts);
				myidx++;
			}
			//类似于压缩在1以内吧
			for(int k=0;k<nn;k++)
				pts[i].neighboursDist[k] *= 10/sumDF;
			//
			if(lvl < pyrLevelsUsed-1 )
			{
				resultSet1.init(ret_index, ret_dist);
				//当前点在上一层的位置
				pt = pt*0.5f-Vec2f(0.25f,0.25f);
				//当前点对应上一层位置的点的Neighbors
				indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());
				//当前点的parent为与它在下一层（分辨率更小）中的估计位置，最近的那个点
				pts[i].parent = ret_index[0];
				pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor);

				assert(ret_index[0]>=0 && ret_index[0] < numPoints[lvl+1]);
			}
			else
			{
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}



	// done.

	for(int i=0;i<pyrLevelsUsed;i++)
		delete indexes[i];
}
}

