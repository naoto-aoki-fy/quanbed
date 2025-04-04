#pragma once
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <tuple>

#include <mpi.h>

auto group_by_host(int const rank, int const size) {

    // 各プロセスでホスト名を取得
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int nameLen;
    MPI_Get_processor_name(hostname, &nameLen);
    std::string my_hostname(hostname, nameLen);

    // rank 0で各プロセスのホスト名を受け取るためのバッファ（固定長）
    std::vector<char> gatheredBuffer;
    if (rank == 0) {
        gatheredBuffer.resize(size * MPI_MAX_PROCESSOR_NAME, '\0');
    }

    // 各プロセスのホスト名をrank 0に集約（固定長文字列）
    MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               (rank == 0 ? gatheredBuffer.data() : nullptr),
               MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
               0, MPI_COMM_WORLD);

    // rank 0側で重複排除とノード番号の付与、さらにノード内のローカルランクを計算する
    std::vector<int> node_numbers; // 各プロセスが所属するノード番号
    std::vector<int> node_local_ranks; // 同一ノード内でのプロセス順（0から開始）
    int node_count = 0;
    if (rank == 0) {
        // 集約された固定長文字列から std::vector<std::string> を作成
        std::vector<std::string> hostnames;
        hostnames.reserve(size);
        for (int i = 0; i < size; i++) {
            const char* ptr = gatheredBuffer.data() + i * MPI_MAX_PROCESSOR_NAME;
            hostnames.push_back(std::string(ptr));
        }

        // ノード番号の付与（重複排除）
        std::unordered_set<std::string> uniqueSet;
        std::vector<std::string> uniqueHosts;
        node_numbers.resize(size, -1);
        for (int i = 0; i < size; i++) {
            const std::string& host = hostnames[i];
            if (uniqueSet.find(host) == uniqueSet.end()) {
                uniqueSet.insert(host);
                uniqueHosts.push_back(host);
                node_numbers[i] = static_cast<int>(uniqueHosts.size()) - 1;
            } else {
                auto it = std::find(uniqueHosts.begin(), uniqueHosts.end(), host);
                node_numbers[i] = static_cast<int>(std::distance(uniqueHosts.begin(), it));
            }
        }
        node_count = static_cast<int>(uniqueHosts.size());

        // 同一ノード内でのプロセス順（ローカルランク）の計算
        node_local_ranks.resize(size, -1);
        std::unordered_map<std::string, int> countMap;
        for (int i = 0; i < size; i++) {
            // 現在のホスト名の出現回数が、そのプロセスのローカルランクになる
            node_local_ranks[i] = countMap[hostnames[i]];
            countMap[hostnames[i]]++;
        }
    }

    // rank 0で決定したノード数を全プロセスにブロードキャスト
    MPI_Bcast(&node_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 各プロセスに自分のノード番号とノード内のローカルランクを通知（scatter）
    int my_node_number = -1;
    int my_node_local_rank = -1;
    MPI_Scatter((rank == 0 ? node_numbers.data() : nullptr), 1, MPI_INT,
                &my_node_number, 1, MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Scatter((rank == 0 ? node_local_ranks.data() : nullptr), 1, MPI_INT,
                &my_node_local_rank, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    // 結果をstderrに出力
    // fprintf(stderr,
    //         "Rank %d on host %s -> assigned node number: %d, local node rank: %d (total nodes: %d)\n",
    //         rank, my_hostname.c_str(), my_node_number, my_node_local_rank, node_count);

    return std::make_tuple(my_hostname, my_node_number, my_node_local_rank, node_count);

}