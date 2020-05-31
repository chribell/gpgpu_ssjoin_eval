#include "gpu.hxx"

#include "cxxopts.hpp"
#include <iostream>

parameters parse(int argc, char* argv[])
{
    try
    {
        // default values
        double threshold = 0.95;
        unsigned int bitmap = 64;

        cxxopts::Options options(argv[0], "GPU set similarity join with bitmap filter");

        options.add_options()
            ("input", "Input dataset file, each line a record", cxxopts::value<std::string>())
            ("foreign-input", "Foreign input dataset file, each line a record", cxxopts::value<std::string>())
            ("threshold", "Similarity threshold", cxxopts::value<double>(threshold))
            ("bitmap", "Bitmap signature size", cxxopts::value<unsigned int>(bitmap))
            ("help", "Print help")
        ;

        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (result.count("threshold")) {
            threshold = result["threshold"].as<double>();
        } else {
            std::cout << "No threshold given, using default value: " << threshold << std::endl;
        }

        if (!result.count("input"))
        {
            std::cerr << "ERROR: No input dataset given! Exiting..." << std::endl;
            exit(1);
        }

        std::string input = result["input"].as<std::string>();

        std::string foreignInput;

        if (result.count("foreign-input")) {
            foreignInput = result["foreign-input"].as<std::string>();
        }

        if (result.count("bitmap")) {
            bitmap = result["bitmap"].as<unsigned int>();
        } else {
            std::cout << "No bitmap signature size given, using default value: " << bitmap << std::endl;
        }

        return { threshold, input, foreignInput, bitmap };

    } catch (const cxxopts::OptionException& e)
    {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[])
{
    auto arguments = parse(argc, argv);
    return gpu(arguments);
}
