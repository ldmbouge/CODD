//
//  RuntimeMonitor.hpp
//  minicpp
//
//  Created by zitoun on 12/17/19.
//  Copyright © 2019 Waldy. All rights reserved.
//

#ifndef RuntimeMonitor_hpp
#define RuntimeMonitor_hpp

#include <chrono>

namespace RuntimeMonitor {
   typedef  std::chrono::time_point<std::chrono::high_resolution_clock> HRClock;
   typedef  std::chrono::time_point<std::chrono::system_clock> SYClock;
   HRClock cputime();
   SYClock wctime();
   HRClock now();
   double elapsedSince(HRClock then);
   long milli(HRClock s,HRClock e);
   double elapsedSeconds(HRClock start, HRClock end);
   double elapsedSeconds(HRClock start);
   unsigned long elapsedSinceMicro(HRClock then);
}

#endif /* RuntimeMonitor_hpp */

