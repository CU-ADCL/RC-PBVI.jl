restore_unregistered(Pkgname, link) = esc(_restore_unregistered(Pkgname, link))

function _restore_unregistered(Pkgname, link)
    s_pkg = string(Pkgname)
    return quote
        try
            using $Pkgname
            println($s_pkg," already installed")
        catch e
            if e isa ArgumentError
                if occursin("not found in current path", e.msg)
                    Pkg.add(url=$link)
                elseif occursin("is required but does not seem to be installed", e.msg)
                    Pkg.rm($s_pkg)
                    Pkg.add(url=$link)
                else
                    println("INVALID ERROR")
                    throw(e)
                end
            else
                println("INVALID ERROR")
                throw(e)
            end
        end
    end
end


@restore_unregistered ConstrainedPOMDPs "https://github.com/qhho/ConstrainedPOMDPs.jl"
@restore_unregistered ConstrainedPOMDPModels "https://github.com/WhiffleFish/ConstrainedPOMDPModels"

@macroexpand @restore_unregistered ConstrainedPOMDPs "https://github.com/qhho/ConstrainedPOMDPs.jl"
