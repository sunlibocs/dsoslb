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


#include "FullSystem/PixelSelector2.h"
 
// 



#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"

namespace dso
{


PixelSelector::PixelSelector(int w, int h)
{
	//w h 640 480 实际过程中
	
	randomPattern = new unsigned char[w*h];
	std::srand(3141592);	// want to be deterministic.
	//　127以内的数
	for(int i=0;i<w*h;i++) randomPattern[i] = rand() & 0xFF;

	currentPotential=3;

	//32 × 32 为一个单元？
	gradHist = new int[100*(1+w/32)*(1+h/32)];
	//每32个一个单元
	ths = new float[(w/32)*(h/32)+100];
	thsSmoothed = new float[(w/32)*(h/32)+100];

	allowFast=false;
	gradHistFrame=0;
}

PixelSelector::~PixelSelector()
{
	delete[] randomPattern;
	delete[] gradHist;
	delete[] ths;
	delete[] thsSmoothed;
}

//要么返回i要么返回90
int computeHistQuantil(int* hist, float below)
{
	//得到值hist[0] 应该是数目
	int th = hist[0]*below+0.5f;
	for(int i=0;i<90;i++)
	{
		th -= hist[i+1];
		if(th<0) return i;
	}
	return 90;
}

//对每一个块，得到其像素梯度出现频率最高的值，并将那个值存储在 ths 中
//得到每个块的梯度与其周围梯度块的均值，并将均值存储在 thsSmoothed 中

void PixelSelector::makeHists(const FrameHessian* const fh)
{
	gradHistFrame = fh;
	//得到金字塔第0 level的 梯度的平方
	float * mapmax0 = fh->absSquaredGrad[0];

	int w = wG[0];
	int h = hG[0];

	int w32 = w/32;
	int h32 = h/32;
	thsStep = w32;

	//对每一个块，得到其像素梯度出现频率最高的值，并将那个值存储在ths中
	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			//得到当前块的起始位置
			float* map0 = mapmax0+32*x+32*y*w;
			int* hist0 = gradHist;// + 50*(x+y*w32);
			//大小为50的空间全部置为0
			memset(hist0,0,sizeof(int)*50);

			//在一个32*32的范围内循环
			for(int j=0;j<32;j++) for(int i=0;i<32;i++)
			{
				int it = i+32*x;
				int jt = j+32*y;
				if(it>w-2 || jt>h-2 || it<1 || jt<1) continue;
				//当前pixel，求平方根
				int g = sqrtf(map0[i+j*w]);
				if(g>48) g=48;
				//将该梯度对应位置的数字+1，也就是为了统计该梯度值的次数？
				hist0[g+1]++;
				//存储计算了多少个梯度
				hist0[0]++;
			}
			//筛选出32 × 32范围内的梯度的值，并且存储在对应的位置
			//这个值要么为90，要么为hist0中的i（也就是梯度值？）
			ths[x+y*w32] = computeHistQuantil(hist0,setting_minGradHistCut) + setting_minGradHistAdd;
		}



	//得到每个块的梯度与其周围梯度块的均值
	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			float sum=0,num=0;
			//以(x,y)为中心的3x3方格的左边的区域
			if(x>0)
			{
				if(y>0) 	{num++; 	sum+=ths[x-1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x-1+(y+1)*w32];}
				num++; sum+=ths[x-1+(y)*w32];
			}
			//以(x,y)为中心的3*3方格的右边的区域
			if(x<w32-1)
			{
				if(y>0) 	{num++; 	sum+=ths[x+1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x+1+(y+1)*w32];}
				num++; sum+=ths[x+1+(y)*w32];
			}
			//(x,y)自己以及它上下的区域
			if(y>0) 	{num++; 	sum+=ths[x+(y-1)*w32];}
			if(y<h32-1) {num++; 	sum+=ths[x+(y+1)*w32];}
			num++; sum+=ths[x+y*w32];

			//得到这个3*3区域的 梯度和/梯度块数（均值） 的平方
			thsSmoothed[x+y*w32] = (sum/num) * (sum/num);

		}



}

//得到用于生成map的，候选的pixel，结果存储在map_out 中
int PixelSelector::makeMaps(
		const FrameHessian* const fh,
		float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
	float numHave=0;
	float numWant=density;
	float quotia;
	int idealPotential = currentPotential;

//	if(setting_pixelSelectionUseFast>0 && allowFast)
//	{
//		memset(map_out, 0, sizeof(float)*wG[0]*hG[0]);
//		std::vector<cv::KeyPoint> pts;
//		cv::Mat img8u(hG[0],wG[0],CV_8U);
//		for(int i=0;i<wG[0]*hG[0];i++)
//		{
//			float v = fh->dI[i][0]*0.8;
//			img8u.at<uchar>(i) = (!std::isfinite(v) || v>255) ? 255 : v;
//		}
//		cv::FAST(img8u, pts, setting_pixelSelectionUseFast, true);
//		for(unsigned int i=0;i<pts.size();i++)
//		{
//			int x = pts[i].pt.x+0.5;
//			int y = pts[i].pt.y+0.5;
//			map_out[x+y*wG[0]]=1;
//			numHave++;
//		}
//
//		printf("FAST selection: got %f / %f!\n", numHave, numWant);
//		quotia = numWant / numHave;
//	}
//	else
	{


		// the number of selected pixels behaves approximately as
		// K / (pot+1)^2, where K is a scene-dependent constant.
		// we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.

		//对每一个块，得到其像素梯度出现频率最高的值，并将那个值存储在 ths 中
		//得到每个块的梯度与其周围梯度块的均值，并将均值存储在 thsSmoothed 中
		if(fh != gradHistFrame) makeHists(fh);

		// select!

		//根据选择pixel， pot参数为currentPotential 
		//选择出像素梯度最大的pixel，结果在map_out中标识，并且使用了图像金字塔
		//Vector3i 中存储了各层选择的pixel的数目
		Eigen::Vector3i n = this->select(fh, map_out,currentPotential, thFactor);

		// sub-select!
		//总共选择出了多少pixel
		numHave = n[0]+n[1]+n[2];
		quotia = numWant / numHave;

		// by default we want to over-sample by 40% just to be sure.
		float K = numHave * (currentPotential+1) * (currentPotential+1);
		idealPotential = sqrtf(K/numWant)-1;	// round down.
		if(idealPotential<1) idealPotential=1;

		//数目不够满足想要的
		//减小pot以探测更多的点
		if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
		{
			//re-sample to get more points!
			// potential needs to be smaller
			if(idealPotential>=currentPotential)
				idealPotential = currentPotential-1;

	//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
	//				100*numHave/(float)(wG[0]*hG[0]),
	//				100*numWant/(float)(wG[0]*hG[0]),
	//				currentPotential,
	//				idealPotential);
			currentPotential = idealPotential;
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);
		}
		//数目太多
		//增大pot以探测更少的点
		else if(recursionsLeft>0 && quotia < 0.25)
		{
			// re-sample to get less points!

			if(idealPotential<=currentPotential)
				idealPotential = currentPotential+1;

			currentPotential = idealPotential;
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);

		}
	}

	int numHaveSub = numHave;
	//如果点的数目太多
	//随机去掉一些
	if(quotia < 0.95)
	{
		int wh=wG[0]*hG[0];
		int rn=0;
		//得到阈值
		unsigned char charTH = 255*quotia;
		for(int i=0;i<wh;i++)
		{
			if(map_out[i] != 0)
			{
				if(randomPattern[rn] > charTH )
				{
					map_out[i]=0;
					numHaveSub--;
				}
				rn++;
			}
		}
	}

//	printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
//			100*numHave/(float)(wG[0]*hG[0]),
//			100*numWant/(float)(wG[0]*hG[0]),
//			currentPotential,
//			idealPotential,
//			100*numHaveSub/(float)(wG[0]*hG[0]));
	currentPotential = idealPotential;


	//这里只是输出细节啊？
	if(plot)
	{
		int w = wG[0];
		int h = hG[0];


		MinimalImageB3 img(w,h);

		for(int i=0;i<w*h;i++)
		{
			float c = fh->dI[i][0]*0.7;
			if(c>255) c=255;
			img.at(i) = Vec3b(c,c,c);
		}
		IOWrap::displayImage("Selector Image", &img);

		for(int y=0; y<h;y++)
			for(int x=0;x<w;x++)
			{
				int i=x+y*w;
				if(map_out[i] == 1)
					img.setPixelCirc(x,y,Vec3b(0,255,0));
				else if(map_out[i] == 2)
					img.setPixelCirc(x,y,Vec3b(255,0,0));
				else if(map_out[i] == 4)
					img.setPixelCirc(x,y,Vec3b(0,0,255));
			}
		IOWrap::displayImage("Selector Pixels", &img);

	}

	return numHaveSub;
}


//参数依次为图像frame， map的状态（大小为w * h）
//选择出像素梯度最大的pixel，结果在map_out中标识，并且使用了图像金字塔
//Vector3i 中存储了各层选择的pixel的数目
Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh,
		float* map_out, int pot, float thFactor)
{

	//fh->dI 用作像素的方向选择
	Eigen::Vector3f const * const map0 = fh->dI;

	float * mapmax0 = fh->absSquaredGrad[0];
	float * mapmax1 = fh->absSquaredGrad[1];
	float * mapmax2 = fh->absSquaredGrad[2];


	int w = wG[0];
	int w1 = wG[1];
	int w2 = wG[2];
	int h = hG[0];

	//十六个方向
	const Vec2f directions[16] = {
	         Vec2f(0,    1.0000),
	         Vec2f(0.3827,    0.9239),
	         Vec2f(0.1951,    0.9808),
	         Vec2f(0.9239,    0.3827),
	         Vec2f(0.7071,    0.7071),
	         Vec2f(0.3827,   -0.9239),
	         Vec2f(0.8315,    0.5556),
	         Vec2f(0.8315,   -0.5556),
	         Vec2f(0.5556,   -0.8315),
	         Vec2f(0.9808,    0.1951),
	         Vec2f(0.9239,   -0.3827),
	         Vec2f(0.7071,   -0.7071),
	         Vec2f(0.5556,    0.8315),
	         Vec2f(0.9808,   -0.1951),
	         Vec2f(1.0000,    0.0000),
	         Vec2f(0.1951,   -0.9808)};
	//这里每次选择的时候会重新置0
	memset(map_out,0,w*h*sizeof(PixelSelectorStatus));



	float dw1 = setting_gradDownweightPerLevel;
	float dw2 = dw1*dw1;


	int n3=0, n2=0, n4=0;
	//每次增加4个plot,从边界0 到h,也就是对整张图都这样做
	for(int y4=0;y4<h;y4+=(4*pot)) for(int x4=0;x4<w;x4+=(4*pot))
	{
		
		int my3 = std::min((4*pot), h-y4);
		int mx3 = std::min((4*pot), w-x4);
		int bestIdx4=-1; float bestVal4=0;
		//得到一个任意的方向 这里pot的值为３
		//randomPattern 一位数组大小为W *H = 640 * 480 很大，不会越界
		//这里因为是使用的×金字塔×第三层，所以方向是这个
		Vec2f dir4 = directions[randomPattern[n2] & 0xF];
		//在上面那个选定的小的范围内（4×pot执行内，这个循环。每次增加量为2×pot）
		for(int y3=0;y3<my3;y3+=(2*pot)) for(int x3=0;x3<mx3;x3+=(2*pot))
		{
			//得到这个block的起点
			int x34 = x3+x4;
			int y34 = y3+y4;
			int my2 = std::min((2*pot), h-y34);
			int mx2 = std::min((2*pot), w-x34);
			int bestIdx3=-1; float bestVal3=0;
			//金子塔的第二层的方向
			//这里因为是使用的×金字塔×第二层，所以方向是这个
			Vec2f dir3 = directions[randomPattern[n2] & 0xF];
			//在上面选定的小范围内（2×pot内执行这个循环，每次增量为pot）
			for(int y2=0;y2<my2;y2+=pot) for(int x2=0;x2<mx2;x2+=pot)
			{
				int x234 = x2+x34;
				int y234 = y2+y34;
				int my1 = std::min(pot, h-y234);
				int mx1 = std::min(pot, w-x234);
				int bestIdx2=-1; float bestVal2=0;
				//方向,在对每一个pot执行的方向是随机的 
				Vec2f dir2 = directions[randomPattern[n2] & 0xF];
				//对pot内的每一个pixel单元执行循环 
				//并且找到那个梯度最大的		
				for(int y1=0;y1<my1;y1+=1) for(int x1=0;x1<mx1;x1+=1)
				{
					assert(x1+x234 < w);
					assert(y1+y234 < h);
					//得到在一维数组中的位置
					int idx = x1+x234 + w*(y1+y234);
					//得到pixel所在的列和行
					int xf = x1+x234;
					int yf = y1+y234;

					if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) continue;

					//xf >> 5相当于 /32， 因此这里是得到pixel所在的块的梯度值
					float pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep];
					//这里乘以了一个新的阈值，
					float pixelTH1 = pixelTH0*dw1;
					//乘以了阈值的平方参数
					float pixelTH2 = pixelTH1*dw2;

					float ag0 = mapmax0[idx];
					//如果梯度大于‘局部块的梯度’ × 因子，则选择
					if(ag0 > pixelTH0*thFactor)
					{
						//得到梯度
						Vec2f ag0d = map0[idx].tail<2>();
						//梯度方向的值，两个向量的点积
						float dirNorm = fabsf((float)(ag0d.dot(dir2)));
						//默认扎个没有用到
						if(!setting_selectDirectionDistribution) dirNorm = ag0;
						//bestVal2 中记录梯度最大的那个
						if(dirNorm > bestVal2)
						{ bestVal2 = dirNorm; bestIdx2 = idx; bestIdx3 = -2; bestIdx4 = -2;}
					}
					//如果在pot内第一层找到了则不再执行后面两层
					if(bestIdx3==-2) continue;

					//这里是否是使用了金字塔？相当于一半+偏移
					float ag1 = mapmax1[(int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1];
					if(ag1 > pixelTH1*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir3)));
						if(!setting_selectDirectionDistribution) dirNorm = ag1;

						if(dirNorm > bestVal3)
						{ bestVal3 = dirNorm; bestIdx3 = idx; bestIdx4 = -2;}
					}
					if(bestIdx4==-2) continue;


					//这里是否使用了第二层金字塔？上面寻找梯度
					float ag2 = mapmax2[(int)(xf*0.25f+0.125) + (int)(yf*0.25f+0.125)*w2];
					if(ag2 > pixelTH2*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir4)));
						if(!setting_selectDirectionDistribution) dirNorm = ag2;

						if(dirNorm > bestVal4)
						{ bestVal4 = dirNorm; bestIdx4 = idx; }
					}
				}
				//如果
				if(bestIdx2>0)
				{
					map_out[bestIdx2] = 1;
					bestVal3 = 1e10;
					n2++;
				}
			}

			if(bestIdx3>0)
			{
				map_out[bestIdx3] = 2;
				bestVal4 = 1e10;
				n3++;
			}
		}

		if(bestIdx4>0)
		{
			map_out[bestIdx4] = 4;
			n4++;
		}
	}


	return Eigen::Vector3i(n2,n3,n4);
}


}

