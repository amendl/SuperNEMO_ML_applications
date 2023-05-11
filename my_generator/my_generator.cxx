
/*
calo type:
1 - main wall
2 - xcalo
3 - gveta


*/

/*
Description of algorithm:
    - x direction: from the front of detector to back
    - y direction: from one side to another
    - z direction: up
    - Size of detector is determined by geiger cell size, which is set to 1.
      Only height of geiger cell cannot be directly calculated from geiger cell size.
      Therfore, we use variable detector height, which should make geometry the same as for demonstrator. 
    - Algorithm iterates over number of events, that should be generated (1st for cycle), then over number of tracks in one event (2nd for cycle).
      Then another two cycles iterate over all geiger cells to determine, wheter there was a hit (brute-force solution)
    - Two possible conditions, whether track goeus through geiger cell: (1) distance from wire (2) whether closest point is inside the box
    - If track is not long enough (<track_min_length), the track is not saved and is recalculated

    TODO geometry of calorimeters
*/

#include <tuple>
#include <variant>
#include <iostream>
#include <memory>
#include <random>

#include <TFile.h>
#include <TTree.h>

const double width = 5000;
const double height = 3100/2;
// http://nile.hep.utexas.edu/DocDB/ut-nemo/docs/0012/001297/001/Tracker%20Module%20Dimensions.pdf

const int       track_min_length = 5;
const double    detector_height = 113.*height/width;


double wire_layer_real(int tracker_layer) {
    return double(tracker_layer)+0.5;
}
double wire_column_real(int tracker_column) {
    return double(tracker_column)+0.5;
}

std::tuple<double,double> projection(double line, std::tuple<double,double> point) {
    double coefficient = (std::get<0>(point)+std::get<1>(point)*line)/(1+line*line);
    return std::make_tuple(coefficient, line*coefficient);
}
double distance(double line, std::tuple<double,double> point) {
    auto p = projection(line,point);
    return sqrt(pow(std::get<0>(p)-std::get<0>(point),2)+pow(std::get<1>(p)-std::get<1>(point),2));
}


void my_generator(int lines, int events, std::string prefix, int file_id) {
    std::cout<<"Options: lines="<<lines<<"; events="<<events<<"; prefix="<<prefix<<"; file_id="<<file_id<<std::endl;

    std::random_device                      rd;
    std::mt19937                            mt(rd());
    std::uniform_real_distribution<double>  foil_y_random_distribution(0.,113.);
    std::uniform_real_distribution<double>  foil_z_random_distribution(-detector_height,detector_height);
    std::uniform_real_distribution<double>  angle_random_distribution(0.01,3.131592); 

    // values into tree
    std::vector<double>                     wirez;
    std::vector<double>                     radius;
    std::vector<int>                        grid_side;
    std::vector<int>                        grid_layer;
    std::vector<int>                        grid_column;
    std::vector<int>                        calo_side;
    std::vector<int>                        calo_column;
    std::vector<int>                        calo_row;
    std::vector<int>                        track_split;

    // helper vairables
    std::vector<double>                     helper_wirez;
    std::vector<double>                     helper_radius;
    std::vector<int>                        helper_grid_side;
    std::vector<int>                        helper_grid_layer;
    std::vector<int>                        helper_grid_column;

    std::cout<<"(Re)Creating file "<<(prefix+std::string("_t")+std::to_string(lines)+std::string("_")+std::to_string(file_id)+".root").c_str()<<std::endl;
    std::unique_ptr<TFile> myFile( TFile::Open((prefix+std::string("_t")+std::to_string(lines)+std::string("_")+std::to_string(file_id)+".root").c_str(),"RECREATE"));
    auto tree = std::make_unique<TTree>("hit_tree", "hit_tree");

    tree->Branch("wirez",                   &wirez);
    tree->Branch("radius",                  &radius);
    tree->Branch("grid_side",               &grid_side);
    tree->Branch("grid_layer",              &grid_layer);
    tree->Branch("grid_column",             &grid_column);
    tree->Branch("calo_side",               &calo_side);
    tree->Branch("calo_column",             &calo_column);
    tree->Branch("calo_row",                &calo_row);
    tree->Branch("track_split",             &track_split);

    // iteration over events
    for(size_t event_id = 0; event_id<events; event_id++) 
    {
        size_t counter = 0;

        wirez                               .clear();
        radius                              .clear();
        grid_side                           .clear();
        grid_layer                          .clear();
        grid_column                         .clear();
        calo_side                           .clear();
        calo_column                         .clear();
        calo_row                            .clear();
        track_split                         .clear();

        // iteration over number of tracks in event
        for (size_t line_id = 0; line_id < lines; line_id++)
        {
            double line_top_projection      = tan(angle_random_distribution(mt));
            double line_front_projection    = tan(angle_random_distribution(mt));
            double x_event                  = foil_y_random_distribution(mt);
            double z_event                  = foil_z_random_distribution(mt); 

            helper_wirez.clear();
            helper_radius.clear();
            helper_grid_side.clear();
            helper_grid_layer.clear();
            helper_grid_column.clear();

            // iteration over geiger cells
            for(size_t tracker_layer = 0; tracker_layer < 9; tracker_layer++)
            {
                // iteration over geiger cells
                for(size_t tracker_column = 0; tracker_column < 113; tracker_column++)
                {
                    auto tracker_point_in_vector_space      = std::make_tuple(wire_layer_real(tracker_layer),wire_column_real(tracker_column)-x_event);
                    auto closest                            = projection(line_top_projection,tracker_point_in_vector_space);
                    double y_closest                        = std::get<0>(closest);
                    double x_closest                        = std::get<1>(closest); // in vector space
                    double tracker_radius                   = distance(line_top_projection, tracker_point_in_vector_space);
                    double z_value                          = z_event + line_front_projection * y_closest;

                    // was there hit?
                    if(abs(x_closest-std::get<1>(tracker_point_in_vector_space))<0.5&& abs(y_closest-std::get<0>(tracker_point_in_vector_space))<0.5 && abs(z_value)< detector_height)
                    //if(tracker_radius<=0.5 && abs(z_value)< detector_height)
                    {
                        helper_wirez                        .push_back(z_value/detector_height*height);
                        helper_radius                       .push_back(tracker_radius*44.);
                        helper_grid_side                    .push_back(1);
                        helper_grid_layer                   .push_back(tracker_layer);
                        helper_grid_column                  .push_back(tracker_column);
                        counter++;
                    }
                }

            } 
            // find hit on wall of the detector
            double x_calo_hit = x_event + line_top_projection*9.;
            double z_calo_hit = z_event + line_front_projection*9.;
            
            // calculate calorimeter hit (https://nemo.lpc-caen.in2p3.fr/wiki/NEMO/SuperNEMO/Calorimeter)
            // TODO geometry of detector
            int calo_hit_column,calo_hit_row;
            if(x_calo_hit<0.) calo_hit_column = 0;
            else if(x_calo_hit > 113.) calo_hit_column = 21;
            else calo_hit_column = int(x_calo_hit/113.*20.+1.);
            if(z_calo_hit<-detector_height) calo_hit_row = 0;
            else if(z_calo_hit > detector_height) calo_hit_row = 14;
            else calo_hit_row = int((detector_height+z_calo_hit)/(2*detector_height)*13.+1.);
            
            // not enough hits in tracker
            if(helper_wirez.size()<track_min_length){
                line_id--;
                counter-=helper_wirez.size();
            }
            // enough hits in tracker, save hit 
            else{
                calo_column.push_back(calo_hit_column);
                calo_row.push_back(calo_hit_row);
                track_split.push_back(counter);
                for(size_t i = 0; i< helper_wirez.size();i++){
                    wirez.          push_back(helper_wirez[i]);
                    radius.         push_back(helper_radius[i]);
                    grid_layer.     push_back(helper_grid_layer[i]);
                    grid_column.    push_back(helper_grid_column[i]);
                }
            }
        }
        tree->Fill();       
    }

    tree->Write();
}