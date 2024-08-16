#include <Eigen/Dense>
#include <iostream>

// Use the Eigen namespace
using namespace Eigen;

double calcLength(Eigen::MatrixXd coord){
    double length = sqrt(pow(coord(1,0)-coord(0,0),2)+pow(coord(1,1)-coord(0,1),2));
    return length;
}

VectorXi getIndices(MatrixXi nodeInd){  
    int n = 2*nodeInd.cols() + 1;
    VectorXi dofs = VectorXi::Zero(n);
    
    dofs(0) = 2*nodeInd(0);
    dofs(1) = 2*nodeInd(0) + 1;
    dofs(2) = 2*nodeInd(1);
    dofs(3) = 2*nodeInd(1) + 1;

    // std::cout << dofs << std::endl;

    return dofs;
}

MatrixXd getCooridanteBeam(MatrixXi e, MatrixXd nodes){
    MatrixXd coord = MatrixXd::Zero(2,2);

    coord.row(0) << nodes(e(0), Eigen::all);
    coord.row(1) << nodes(e(1), Eigen::all);

    //std::cout << coord << std::endl;

    return coord;
}

MatrixXd assembleMatrix(MatrixXd K, MatrixXd Kloc, MatrixXi indices){
    // std::cout << Kloc.rows() << std::endl;
    // std::cout << indices << std::endl;
    // std::cout << "-----" << std::endl;
    for(int i = 0; i < Kloc.rows(); i++){
        for(int j = 0; j < Kloc.cols(); j++){
            // std::cout << "(" << i << "," << j << ")" << std::endl;
            K(indices(i), indices(j)) = K(indices(i), indices(j)) + Kloc(i, j);
        }
    }
    return K;
}

VectorXd assembleVector(VectorXd fb, VectorXd floc, VectorXi indices){
    for(int i = 0; i < floc.rows(); i++){
        fb(indices(i)) = fb(indices(i)) + floc(i);
    }
    return fb;
}

MatrixXd buildLocalK(MatrixXd coord, double E, double G, double As, double I){
    MatrixXd K = MatrixXd::Zero(4, 4);

    double L = calcLength(coord);

    K(0,0) = G*As/L;
    K(0,1) = G*As/2;
    K(0,2) = -G*As/L;
    K(0,3) = G*As/2;

    K(1,0) = K(0,1);
    K(1,1) = E*I/L + G*As*L/4;
    K(1,2) = -G*As/2;
    K(1,3) = -E*I/L + G*As*L/4;

    K(2,0) = K(0,2);
    K(2,1) = K(1,2);
    K(2,2) = G*As/L;
    K(2,3) = -G*As/2;

    K(3,0) = K(0,3);
    K(3,1) = K(1,3);
    K(3,2) = K(2,3);
    K(3,3) = E*I/L + G*As*L/4;

    return K;

}

MatrixXd buildGlobalK(MatrixXd nodes, MatrixXi elements, double E, double G, double As, double I){
    int ne = elements.rows();
    int ndof = 2*nodes.rows();
    std::cout << "Number of degrees of freedom: " << ndof << std::endl;
    MatrixXd K = MatrixXd::Zero(ndof, ndof);

    // degrees of freedom
    VectorXi e_dofs;

    for(int i=0; i<ne; i++){
        MatrixXi e = elements.row(i);
        e_dofs = getIndices(e);

        
        MatrixXd coord = getCooridanteBeam(e, nodes);
        MatrixXd Kloc = buildLocalK(coord, E, G, As, I);
        std::cout << "Kloc: \n" << Kloc << std::endl;
        K = assembleMatrix(K, Kloc, e_dofs);
    }

    // std::cout << K << std::endl;
    // std::cout << "-----" << std::endl;

    return K;
}

MatrixXd applyDBCMatrix(MatrixXd K, MatrixXi supp){
    int n_supp = supp.rows();
    int ind; // dof index

    for(int i=0; i<n_supp; i++){
        if(supp(i, 1) == 1){
            ind = 3*supp(i,0);
            K.row(ind).setZero();
            K.col(ind).setZero();
            K(ind,ind) = 1;
        }

        if(supp(i,2) == 1){
            ind = 3*supp(i,0) + 1;
            K.row(ind).setZero();
            K.col(ind).setZero();
            K(ind,ind) = 1;
        }

        if(supp(i,3) == 1){
            ind = 3*supp(i,0) + 2;
            K.row(ind).setZero();
            K.col(ind).setZero();
            K(ind,ind) = 1;
        }
    }
    
    return K;
}

VectorXd applyDBCVec(VectorXd R, MatrixXd supp){
    int n_supp = supp.rows();
    int ind; // dof index

    for(int i=0; i<n_supp; i++){
        if(supp(i, 1) == 1){
            ind = 3*supp(i,0);
            R(ind) = 0;
        }

        if(supp(i,2) == 1){
            ind = 3*supp(i,0) + 1;
            R(ind) = 0;
        }

        if(supp(i,3) == 1){
            ind = 3*supp(i,0) + 2;
            R(ind) = 0;
        }
    }
    return R;
}

namespace mesh {

    class beam {
    public:
        Eigen::MatrixXd nodes;
        Eigen::MatrixXi elements;

        void horizontalBarDisc(double bar_length, int num_elements);
    };

    void beam::horizontalBarDisc(double bar_length, int num_elements) {
        // Initialize nodes array with zeros
        nodes = Eigen::MatrixXd::Zero(num_elements + 1, 2);

        // Initialize elements array with zeros
        elements = Eigen::MatrixXi::Zero(num_elements, 2);

        // Mesh increment
        double inc = bar_length / num_elements;
        double coord = inc;

        // Initialize index controller
        int i = 1;

        while (coord <= bar_length) {
            nodes(i, 0) = coord;
            i += 1;
            coord += inc;
        }

        for (int i = 0; i < num_elements; i++) {
            elements(i, 0) = i;
            elements(i, 1) = i + 1;
        }
    }

}


int main(){
    mesh::beam beam;
    beam.horizontalBarDisc(100.0, 5);

    double E = 1e+7;
    double G = 5e+6;
    double As = 1;
    double I = 1.0/12.0;

    Eigen::MatrixXi supp = Eigen::MatrixXi::Zero(1,4);
    supp << 0, 1, 1, 0;

    VectorXd f = VectorXd::Zero(2*beam.nodes.rows());
    f(f.size()-2) = -4;
    

    MatrixXd K = buildGlobalK(beam.nodes, beam.elements, E, G, As, I);
    MatrixXd K_ = applyDBCMatrix(K, supp);

    Eigen::VectorXd uh;
    uh = K_.ldlt().solve(f);

    std::cout << "uh:\n" << uh << std::endl;
    std::cout << "f: \n" << f << std::endl;
    std::cout << "K: \n" << K_ << std::endl;

    return 0;
}

