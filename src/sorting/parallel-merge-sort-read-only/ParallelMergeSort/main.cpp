
#include "src/includes.h"

#include "src/mls.h"

#include "ARAPWrapper.h"

void def_weights_uniform(vector<ublas::vector<float> > &W, vector<vector<int> > &ne)
{
	W.resize(ne.size());

	for ( unsigned i = 0 ; i < ne.size() ; i++ ) {
		W[i] = ublas::scalar_vector<float>(ne[i].size(),1); // Laplacian weights: 1 for 1-ring ne
	}
}

void update_rhs(ublas::vector<float> &rhs,
				ublas::matrix<float> &srcPts,
				vector<vector<int> > &ne, 
				vector<ublas::vector<float> > &W,
				vector<ublas::matrix<float> > &Rs)
{
	int N = (int)srcPts.size1();

	for ( int i = 0 ; i < N ; i++ ) {
		ublas::vector<float> b(3);
		b.clear();
		for ( int n = 0 ; n < (int)ne[i].size() ; n++ ) {
			int j = ne[i][n];
			ublas::vector<float> p = row(srcPts,i) - row(srcPts,j);
			ublas::matrix<float> R = Rs[i]+Rs[j];
			ublas::vector<float> Rp = ublas::prod(R, p);
			b += W[i](n)/2.0 * Rp;
		}
		for ( int c = 0 ; c < 3 ; c++ )
			rhs(N*c+i) += b(c);
	}
}

void get_rotations_and_update_rhs(ublas::vector<float> &rhs,
								  ublas::matrix<float> &srcPts, 
								  ublas::matrix<float> &pts, 
								  vector<vector<int> > &ne,
								  vector<ublas::vector<float> > &W,
								  bool bWithScaling = false)
{
	vector<ublas::matrix<float> > Rs;
	get_rotations(srcPts, pts, ne, Rs, W, 1);
	update_rhs(rhs, srcPts, ne, W, Rs);
}

void main()
{
	const int N = 3;//10000; // Number of points
	const int nIter = 100; // Num. Iterations

	ublas::matrix<float> srcPts(N,3);
	ublas::matrix<float> pts(N,3); 
	vector<vector<int> > ne;
	vector<ublas::vector<float> > W;
	ublas::vector<float> rhs_init(3*N);

	srand(time(NULL));

	// init
	// set some point values
	for  ( int i = 0 ; i < N ; i++ )
		for  ( int c = 0 ; c < 3 ; c++ ) {
			srcPts(i,c) = rand();
			rhs_init(i+c*N) = rand();
		}

	// set arbitrary neighbors
	ne.resize(N);
	for  ( int i = 0 ; i < N ; i++ )
		for  ( int j = -3 ; j < 3 ; j++ )
			if ( j != 0 )
				ne[i].push_back((j+i+N)%N);
	def_weights_uniform(W, ne);

	ublas::vector<float> rhs = rhs_init;

	for  ( int i = 0 ; i < N ; i++ )
		for  ( int j = 0 ; j < 3 ; j++ )
			pts(i,j) = rand();


	ARAPWrapper::get_rotations_and_update_rhs(rhs, srcPts, pts, ne, W, 1);

	//// iterations
	//for ( int it = 0 ; it < nIter ; it++ ) {
	//	// set some points (simulate some calculation I do on the points)
	//	for  ( int i = 0 ; i < N ; i++ )
	//		for  ( int j = 0 ; j < 3 ; j++ )
	//			pts(i,j) = rand();

	//	ublas::vector<float> rhs = rhs_init;
	//	get_rotations_and_update_rhs(rhs, srcPts, pts, ne, W, 1);
	//}

}
